# spot-voice

Hands-free, voice-commanded operation of a Boston Dynamics Spot for facility
inspection walkthroughs.

You talk into a wireless lav mic. Speech is transcribed **locally** on the
laptop. Safety words fire hardcoded robot commands immediately, before anything
else happens. Everything else goes to Claude, which drives the robot through the
official Spot SDK and answers out loud through Spot's own speaker.

Internal project. Private repo, no telemetry, no web UI, no external tool
servers, no MCP anywhere — just the Anthropic Messages API and `bosdyn-client`.

---

## Contents

1. [How it works](#how-it-works)
2. [Safety model](#safety-model)
3. [Setup on Windows 11](#setup-on-windows-11)
4. [Picking the microphone](#picking-the-microphone)
5. [Configuration](#configuration)
6. [Running in mock mode](#running-in-mock-mode)
7. [The dual-network setup](#the-dual-network-setup)
8. [Staged rollout on the real robot](#staged-rollout-on-the-real-robot)
9. [What Claude can do](#what-claude-can-do)
10. [Follow-me](#follow-me)
11. [Privacy: what leaves the laptop](#privacy-what-leaves-the-laptop)
12. [Project layout](#project-layout)
13. [Tests](#tests)
14. [Troubleshooting](#troubleshooting)

---

## How it works

```
your voice
  -> sounddevice capture, 16 kHz mono
  -> VAD segmentation (webrtcvad, energy-gate fallback)
  -> faster-whisper, locally, int8
  -> reflex check
       |  match    -> hardcoded robot command, immediately   (no network)
       |  no match -> Claude tool-use loop -> robot layer -> Spot
  -> reply text -> TTS -> Spot's speaker
```

Two lanes, deliberately separate:

**The reflex lane** matches the transcript against a small set of safety phrases
before any API call. On a hit it runs a hardcoded action and the utterance never
reaches Claude. It has no dependency on the Anthropic SDK, the internet, or
anything else that can be slow or absent — pull the network cable and "stop"
still stops the robot.

**The Claude lane** is a plain Anthropic Messages API tool-use loop. Claude gets
thirteen tools, calls them, reads the results (including camera images), and
replies in one or two sentences. That reply is spoken aloud.

---

## Safety model

- **Spot's own safety systems are never touched.** Obstacle avoidance,
  self-righting and stair handling stay at their factory defaults. There is no
  code path that changes an obstacle-padding parameter, and no tool exposes one.
  A test (`test_no_tool_exposes_a_safety_override`) fails the build if one is
  ever added.
- **Hard velocity caps live in code, not config.** `|v_x| <= 0.6 m/s`,
  `|v_y| <= 0.4 m/s`, `|v_rot| <= 0.8 rad/s`. Every velocity that reaches the SDK
  passes through `clamp_velocity()`. Nothing in `.env` can raise them and no tool
  argument can exceed them.
- **Dead-man's switch.** Velocity commands carry `end_time_secs = now + 0.6 s`
  and are re-issued every 0.25 s by an active loop. If the program crashes, the
  laptop sleeps, or wifi drops mid-walk, the last command expires and Spot stops
  itself.
- **Software e-stop.** A no-GUI e-stop endpoint checks in every few seconds. If
  check-ins stop, the robot cuts motor power on its own. Releasing the e-stop
  (`allow`) happens once, at startup, from this program — it is **not** exposed
  as a Claude tool.
- **The reflex "stop" is a safe stop, not a power cut.** It cancels the active
  command and any GraphNav route and settles into a stable stand.
  `settle_then_cut` is reserved for a real emergency power cut.
- **The physical e-stop on the tablet is always the ultimate authority.** Keep a
  human on it for every session on the real robot.
- **The matcher deliberately over-triggers.** "don't stop" stops the robot. A
  spurious stop is an inconvenience; a missed stop is a safety incident.

### Safety words

| Say | Action |
|---|---|
| stop, freeze, halt, whoa, hold up, stop it, stop now, emergency stop | cancel everything, safe stop |
| stop following, stop following me, quit following, stay there | kill follow-me, then settle |
| sit, sit down, lie down, take a seat | sit |

Matching is fuzzy, so `stopp`, `stopped` and `sitt down` all register — that is
the shape of error speech-to-text actually makes.

### One known gap: the mic is gated while Spot is talking

By default (`MUTE_WHILE_SPEAKING=true`) the microphone is muted while Spot is
speaking, so it does not transcribe its own voice and talk to itself. For the two
or three seconds a reply lasts, a spoken safety word is not heard. **The tablet
e-stop is the answer to that gap.** If your mic is far enough from the speaker
(laptop output, operator wearing the lav) you can set it to `false` and get
uninterrupted listening.

---

## Setup on Windows 11

Python 3.11 or 3.12.

```powershell
# 1. Get the code and open a terminal in it
cd C:\Users\<you>\spot-voice

# 2. Create and activate a virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure
copy .env.example .env
notepad .env
```

If PowerShell refuses to run the activate script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Notes on individual packages

- **`bosdyn-client` / `bosdyn-api` / `bosdyn-core`** — the official Spot SDK.
  Only imported when `MOCK_ROBOT=false`; mock mode runs fine without them.
- **`webrtcvad-wheels`** — prebuilt WebRTC VAD, no compiler needed. If it fails
  to install, the program falls back to an energy-based voice detector
  automatically. It is a little less precise but perfectly usable.
- **`simpleaudio`** — needs a build toolchain on some Python versions. If it
  fails, skip it: the program falls back to Windows' built-in `winsound`.
- **`faster-whisper`** downloads its model on first run (~150 MB for `base`).
  Do that once while you have decent internet, not on the demo floor.
- **`ultralytics`** downloads `yolov8n.pt` (~6 MB) the first time follow-me runs
  on the real robot. Same advice.

To regenerate the placeholder camera image used by mock mode you also need
Pillow (`pip install -r requirements-dev.txt`), then
`python scripts/make_test_scene.py`.

---

## Picking the microphone

The Hollyland Lark M1 receiver appears on Windows as a generic input device,
usually **"USB Audio Device"**, and its index moves between reboots — so the
device is selected by a case-insensitive substring of its name, not by index.

```powershell
python -m spot_voice --list-devices
```

Pick anything distinctive from the name column and put it in `.env`:

```
MIC_DEVICE_NAME=USB Audio
```

Leave it blank to use the Windows default input. If the substring matches
nothing, startup fails with the full device list printed, so you can see what to
type instead.

---

## Configuration

Everything lives in `.env`. Nothing is hardcoded and `.env` is gitignored.

| Variable | What it does |
|---|---|
| `MOCK_ROBOT` | `true` simulates the robot; `false` drives a real Spot |
| `SPOT_IP` | Spot's address on its own wifi |
| `BOSDYN_CLIENT_USERNAME` / `BOSDYN_CLIENT_PASSWORD` | read by the Spot SDK itself; this project never touches them |
| `GRAPH_PATH` | folder containing `graph`, `waypoint_snapshots/`, `edge_snapshots/` |
| `DOCK_ID` | dock fiducial id; blank if there is no dock |
| `ANTHROPIC_API_KEY` | your key |
| `ANTHROPIC_MODEL` | default `claude-sonnet-4-6` |
| `MIC_DEVICE_NAME` | substring of the input device name |
| `WHISPER_MODEL` | `tiny` / `base` / `small` |
| `STT_LANGUAGE` | forced decode language, default `en` |
| `TTS_ENGINE` | `edge` (online, better voice) or `offline` (local, private) |
| `TTS_VOICE` | edge-tts voice name |
| `AUDIO_OUT` | `robot` (Spot CAM speaker) or `laptop` |
| `MUTE_WHILE_SPEAKING` | gate the mic while Spot talks; default `true` |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` |

### A note on the model

`claude-sonnet-4-6` is the configured default: fast enough for conversation and
strong at tool use. `claude-sonnet-5` and `claude-opus-5` are the
current-generation equivalents and are drop-in replacements — change the one line
in `.env` if you want more capability, particularly for describing what the
camera sees.

The code sends no `thinking` parameter. For a voice loop, latency is what the
operator feels, and replies are one or two sentences; omitting it gives the
fastest first token and is valid on every current model, so swapping the model id
never requires a code change.

### GraphNav map layout

`GRAPH_PATH` should point at the folder that **directly contains** the graph
files, which is usually named `downloaded_graph`:

```
C:\maps\my_facility\downloaded_graph\
    graph
    waypoint_snapshots\
    edge_snapshots\
```

---

## Running in mock mode

Mock mode is first-class and stays that way. Every tool, the reflex lane, the
Claude loop, follow-me and TTS all run for real — only the SDK calls are replaced
with logging and a small state machine. The whole experience runs at a desk with
nothing but a microphone.

```powershell
# .env: MOCK_ROBOT=true, AUDIO_OUT=laptop

# Full voice loop
python -m spot_voice

# Type instead of talking (no microphone, no Whisper model download)
python -m spot_voice --text

# One command and exit
python -m spot_voice --say "go to the control panel and tell me what you see"
```

Things worth trying in mock mode:

- `stand`, `sit down`, `stop` — watch the reflex latency printed in ms
- `what places do you know` — reads the waypoint list
- `go to the control panel and look around` — the inspection flow: navigate,
  capture an image, describe it
- `follow me` then `stop following` — the follow-me thread with a simulated
  person that moves left and right and occasionally disappears, so the "I lost
  you" path gets exercised too

`capture_image` returns a bundled synthetic facility scene
(`spot_voice/assets/test_scene.jpg`). Drop a real 640x480 JPEG in its place to
have Claude describe your actual site.

---

## The dual-network setup

**First, check which network your Spot is actually on — it decides whether any
of this applies.**

| Spot's address | What it means | What you need |
|---|---|---|
| `192.168.208.117` | The EDGE facility LAN. This is what the existing internal platform connects to. | If the laptop is on that network and it has internet, **one interface does everything**. Skip the rest of this section. |
| `192.168.80.3` | Spot's own access point. No internet on it. | You need the two-interface setup below. |

Ping it before you plan anything:

```bash
ping 192.168.208.117
```

The rest of this section is for the access-point case only.

The laptop needs **two networks at once**:

1. **Spot's wifi** — to reach the robot at `SPOT_IP`. Spot's access point has no
   internet.
2. **An internet path** — for the Anthropic API. Phone USB tethering or an
   ethernet dongle both work.

Windows will route based on interface metric. In practice:

- Join Spot's wifi on the wifi adapter.
- Plug in the phone (USB tethering) or ethernet for internet.
- Check that both work before you start:

```powershell
ping 192.168.80.3          # or whatever SPOT_IP is
curl https://api.anthropic.com/v1/models -H "x-api-key: %ANTHROPIC_API_KEY%" -H "anthropic-version: 2023-06-01"
```

If the Anthropic call fails while Spot pings fine, Windows is routing everything
over the Spot adapter. Fix it by raising the metric on the wifi adapter so it is
*less* preferred for the default route:

```powershell
Get-NetIPInterface -AddressFamily IPv4 | Format-Table ifIndex, InterfaceAlias, InterfaceMetric
Set-NetIPInterface -InterfaceAlias "Wi-Fi" -InterfaceMetric 60
```

Spot traffic still finds the robot because that subnet is directly connected;
everything else goes out over the tether.

**The program is built for this link being slow.** The Anthropic request timeout
is 60 seconds with two retries, and a connection failure produces a spoken "I
can't reach my language service right now, but safety commands still work"
rather than a crash. The reflex lane is entirely unaffected by internet quality.

---

## Staged rollout on the real robot

Do not skip stages. Each one adds exactly one new failure mode.

### Stage 1 — Mock at your desk

`MOCK_ROBOT=true`, `AUDIO_OUT=laptop`. No robot involved.

- [ ] Mic is detected and transcripts are accurate for your voice and accent
- [ ] "stop" reflex fires in well under 300 ms (printed after every reflex)
- [ ] Claude answers, calls tools, and stays to one or two sentences
- [ ] Follow-me starts and stops on command
- [ ] Speech plays through the laptop at a volume you can hear across a room

### Stage 2 — Real robot, sit and stand only

`MOCK_ROBOT=false`, `AUDIO_OUT=laptop` first (add the robot speaker in stage 3).

- [ ] **A human is on the tablet e-stop for the whole session**
- [ ] Clear floor space, nobody within Spot's reach
- [ ] Robot undocked manually, or `undock` tested first
- [ ] Say "stand", then "sit". Confirm each completes
- [ ] Say "stop" while it is standing up. Confirm it settles
- [ ] Check `get_status` reports sensible battery and e-stop state

Stop here and fix anything odd before moving on.

### Stage 3 — Short moves in open space

- [ ] At least 5 m of clear floor in every direction
- [ ] Human on the tablet e-stop
- [ ] "walk forward one metre" — confirm the distance is roughly right
- [ ] "turn left" / "turn right"
- [ ] Say "stop" *mid-move*. Confirm it stops promptly
- [ ] Pull the laptop's wifi mid-move and confirm the robot stops itself within
      about a second — this is the dead-man's switch, and it is worth seeing once
- [ ] Switch `AUDIO_OUT=robot` and confirm speech comes out of Spot

### Stage 4 — Waypoint navigation

- [ ] Map uploaded, `GRAPH_PATH` correct
- [ ] Spot can see a fiducial from its starting position
- [ ] "what places do you know" returns the expected names
- [ ] "go to <waypoint>" for a short, simple route first
- [ ] Say "stop" mid-route. Confirm the route is abandoned
- [ ] Then the full inspection flow: "go to <waypoint>, look around and tell me
      what you see"

### Stage 5 — Follow-me

- [ ] Wide, open, uncluttered area
- [ ] Human on the tablet e-stop, and a second person watching the robot
- [ ] `TARGET_BBOX_HEIGHT_FRACTION` calibrated (see below)
- [ ] "follow me" — walk slowly, in a straight line, first
- [ ] "stop" and "stop following" both kill it instantly
- [ ] Step out of frame and confirm "I lost you" after about two seconds
- [ ] Only then try turns and corners

---

## What Claude can do

Thirteen tools. Every one returns `{ok, message}` and never raises into the
model — a failure comes back as data, and the system prompt tells Claude to speak
the message it was given.

| Tool | What it does |
|---|---|
| `stand` | power on if needed, stand up |
| `sit` | sit down |
| `move` | bounded timed segment: forward / back / left / right / turn_left / turn_right |
| `navigate_to` | GraphNav walk to a named waypoint |
| `list_waypoints` | the named places on the map |
| `start_follow` / `stop_follow` | follow-me |
| `capture_image` | one upright JPEG from front / left / right, returned to Claude as an image so it can actually see it |
| `get_status` | battery, motor power, lease, e-stop, localization, dock |
| `dock` / `undock` | charging dock |
| `speak` | say something mid-task |
| `stop_all` | cancel everything, safe stop |

There is deliberately **no** tool for releasing the e-stop, changing
obstacle-avoidance parameters, or otherwise weakening a safety system.

### The inspection flow

The whole point of the tool set is that this works as one natural request:

> "Go to the compressor room, look around and tell me what you see."

Claude chains `navigate_to` -> `capture_image` -> describes the image in a
sentence or two -> that reply is spoken through Spot's speaker.

### Prompt caching

The system prompt and the tool definitions are frozen for the whole session and
each carries one cache breakpoint. A voice session is many short turns over the
same large prefix, which is exactly the case caching is for. Watch it working:

```
LOG_LEVEL=INFO
# anthropic in=… out=… cache_write=… cache_read=… stop=…
```

If `cache_read` stays at zero across turns, something is changing in the prefix.
The waypoint list is fetched once at startup for that reason.

---

## Follow-me

`start_follow` spawns a background thread. Claude only starts and stops it; it
never steers.

The loop runs at 8 Hz: front camera frame -> YOLOv8-nano person detection (CPU is
fine) -> pick the largest, most centred person -> P-control with deadbands ->
capped velocity command with the 0.6 s expiry.

- Yaw rate comes from horizontal offset from frame centre.
- Forward speed comes from apparent size versus a target standoff of about 1.5 m.
- Too close: it holds position rather than backing up blind.
- Person lost for more than two seconds: stop and say "I lost you."
- `stop_follow`, the spoken word "stop", and `sit` all kill the thread instantly.

**Spot's obstacle avoidance is the safety net and stays on at factory defaults.**
Obstacle padding is never reduced.

### The one number to calibrate

`TARGET_BBOX_HEIGHT_FRACTION` in `spot_voice/robot/follow.py` (default `0.55`) is
the fraction of frame height a person occupies at the desired standoff. To
calibrate: stand 1.5 m in front of Spot, run follow-me with `LOG_LEVEL=DEBUG`,
and read the observed box height. Adjust and re-test.

In mock mode a simulated person is used instead of YOLO, so follow-me is fully
exercisable at a desk — including the lost-target path.

---

## Privacy: what leaves the laptop

| Data | Where it goes |
|---|---|
| Microphone audio | **Nowhere.** Transcription is local (faster-whisper). No cloud STT. |
| Transcripts and Claude's replies | Anthropic API, over HTTPS. |
| Camera images | Anthropic API, only when `capture_image` is called. |
| Reply text, `TTS_ENGINE=edge` | **Microsoft's Edge TTS servers.** |
| Reply text, `TTS_ENGINE=offline` | Nowhere. Local Windows SAPI5 voice. |
| Anything else | Nowhere. No telemetry, no analytics, no external tool servers. |

Set `TTS_ENGINE=offline` if reply text must not reach Microsoft. The voice is
less pleasant; nothing else changes.

---

## Project layout

```
spot_voice/
  main.py            entry point, wiring, CLI
  config.py          .env loading and validation
  audio/
    devices.py       list inputs, select by substring
    vad.py           voice activity detection, utterance segmentation
    stt.py           faster-whisper wrapper
    listener.py      capture thread + worker thread
  safety/
    reflex.py        the safety-word matcher and its hardcoded actions
  brain/
    prompts.py       the system prompt
    tools.py         the thirteen tool schemas
    dispatcher.py    the single choke point for every tool call
    agent.py         the Anthropic Messages API tool-use loop
  robot/
    base.py          the interface mock and real both implement
    limits.py        hard velocity caps
    motion.py        move planning and waypoint name matching, shared by both
    errors.py        SDK exceptions -> speakable sentences
    estop.py         no-GUI software e-stop
    graphnav.py      map upload, fiducial localization, navigation
    spot_client.py   the real SDK layer
    mock.py          the simulated robot
    follow.py        follow-me thread and P-controller
  tts/
    engines.py       edge-tts and pyttsx3, both producing 16 kHz mono WAV
    speaker.py       synthesis + routing to robot or laptop
  assets/
    test_scene.jpg   placeholder camera frame for mock mode
scripts/
  make_test_scene.py regenerate that image
tests/
```

---

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

209 tests, no robot and no API key required. They cover:

- the reflex matcher: stop words, transcription slips, and the false positives it
  must *not* fire on
- the tool dispatcher: every tool returns `{ok, message}`, malformed model output
  never raises, a raising robot becomes a failed result
- velocity capping, including NaN and infinity
- tool schemas, including a guard that no safety-override tool has been added
- the follow-me controller's target selection and control law
- the Anthropic tool-use loop end to end against a scripted fake client:
  parallel tool calls, image attachment, refusals, connection failure, abort, and
  the runaway-loop cap
- conversation trimming, which must never leave an orphaned `tool_result`
- that the mock and the real client stay signature-identical, so mock mode keeps
  predicting what the robot will actually do

Where the Spot SDK is installed, an extra set checks that every `bosdyn` symbol
the real client uses still resolves — so an SDK rename shows up here rather than
on the robot. Those tests skip cleanly on a mock-only machine.

---

## Troubleshooting

**"No input device matching …"**
Run `python -m spot_voice --list-devices` and copy a distinctive part of the name
into `MIC_DEVICE_NAME`.

**Transcripts are wrong or empty**
Try `WHISPER_MODEL=small`. Check the mic is not muted at the OS level and that
the lav is close to your mouth. `LOG_LEVEL=DEBUG` prints the audio length of each
utterance, so you can tell whether segmentation or transcription is at fault.

**Spot talks to itself**
`MUTE_WHILE_SPEAKING=true` (the default) prevents this. If it is off, turn it on
or move the mic away from the speaker.

**"I can't reach my language service right now"**
The Anthropic call failed. Almost always the dual-network routing — see
[The dual-network setup](#the-dual-network-setup). The robot is unaffected and
safety words keep working.

**"Spot is claimed by another controller"**
Another client holds the lease. Close the tablet app or the other session. This
program takes stale leases deliberately, but not live ones.

**"Spot is still booting"**
Wait for the robot to finish starting and try again.

**"I can't localize. I need to see a fiducial marker."**
GraphNav needs to see a fiducial to fix its position. Walk Spot until one is in
view, or start it from a known spot near a marker.

**edge-tts fails, or there is no audio**
edge-tts needs internet and an MP3 decoder. `miniaudio` provides one with no
external binary; `imageio-ffmpeg` is an alternative. If neither is present the
program falls back to the offline voice automatically and says so. Setting
`TTS_ENGINE=offline` skips all of it.

**Follow-me will not start**
`ultralytics` is missing, or its first-run model download failed. Check internet
and run `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` once.

**Unicode errors in the console**
Shouldn't happen — all output is ASCII and unencodable characters in a reply are
replaced rather than raised. If you see one, please report it with the reply text.
