# The two viewers: MuJoCo Studio and viser

linker-sim shows a station in one of two viewers: **MuJoCo Studio**, the default, or
**viser**, the backup. Both are installed. Both are web pages: the simulation runs on this
computer's CPU, and your browser draws the scene. An interface is written once as a
`viewer.Panel` and runs in either one:

```bash
uv run linker-sim play                   # Studio
uv run linker-sim play --viewer viser    # viser
```

```python
from linker_sim import viewer
from linker_sim.viewer import Panel
from linker_sim.workstation import Station

station = Station()
panel = Panel("My lesson")
panel.button("Reset", station.reset, key="r")
viewer.run(station, panel)                  # Studio
viewer.run(station, panel, viewer="viser")  # the same panel in viser
```

## What differs

| | MuJoCo Studio | viser |
| --- | --- | --- |
| Drawing | MuJoCo's own renderer, in the browser (WebAssembly and WebGL) | mjviser turns the model into viser meshes and moves them every frame |
| Buttons, lists, sliders, text, markers, keys | Yes | Yes |
| **Held keys** (`Panel.hold`) | Yes: a key moves something for as long as it is held | **Presses only.** Each press counts as the key held for 0.1 s, and the keyboard's auto-repeat extends it. So a held key moves a step, then continues once the repeat starts (about 0.5 s) |
| **Drag handle** (`Panel.handle`) | **No.** Offer the same through keys or sliders | Yes |
| Several browsers on one page | **One controls** the panel; others watch with their own camera, and a page can take control ("Steal control") | Every page shows the panel |
| Camera keys | None | W, A, S, D, Q, E and the arrow keys move the camera, so panels leave those letters free |
| What each page downloads | The whole compiled model: about 290 MB for the playground, 345 MB for the gallery | About 16 MB of meshes |
| Without a GPU in the browser's computer | Unusable: the page draws below 2 frames per second, and its keys and buttons wait for the frames (seconds) | Slow to look at, but its panel and keys stay responsive |

**Choose viser** when Studio doesn't work on a computer or browser, when you need the
drag handle, when pages are opened over a slow network, or when a MuJoCo update breaks
Studio before linker-sim is fixed. Every feature of the playground and the gallery works
in both viewers, apart from the two marked **No** above.

## Speed

Measured on our development computer (x86_64 Linux) with a browser on the same computer
drawing in hardware:

- both viewers draw at 60 frames per second, and the page is ready 3–4 s after opening
- the simulation runs at real time (1.00×) in both
- the spare capacity (how much faster it could run) is shown on every panel's last line:
  in the gallery 8.6× idle and 7.0× while grasping in Studio, and 5.2× and 4.2× in viser
  (viser's per-frame mesh update costs more)

The browser drawing LW-v0's 1.5 million triangles is the heavy part, not the simulation.
A browser that renders in software, with no usable GPU driver as in virtual machines and
some remote desktops, draws Studio at 0.1–1.6 frames per second. Check that the browser
uses the GPU: in Chrome, `chrome://gpu` shows "Hardware accelerated" for WebGL.

## Network: this computer only

Both viewers listen on `127.0.0.1` only, so only this computer can open the page. Studio
serves plain HTTP with no login, and its own docs call it a development tool. So
linker-sim never exposes it to the network.

- **From another computer of yours:** forward the port over SSH, e.g.
  `ssh -L 8080:127.0.0.1:8080 <computer>`, then open `http://127.0.0.1:8080/` locally.
- **For your users:** put an HTTPS reverse proxy that you run in front of the page.
  Browsers only start the viewer from HTTPS or from localhost. Authentication is the
  proxy's job.

## Inside your own web page

Both pages work inside an `<iframe>` of a page from another origin. Clicks and keys reach
the simulation, with no console errors.

- Studio's page isolates itself: it sends `Cross-Origin-Opener-Policy: same-origin`,
  `Cross-Origin-Embedder-Policy: require-corp` and
  `Cross-Origin-Resource-Policy: cross-origin`, and no `X-Frame-Options`. Your page must
  allow the frame.
- viser sends none of these headers.
- In Studio one page controls and the others watch, an embedded page included. Plan for
  it: one simulation per student, or one that a teacher drives while students watch.

## When a Panel is not enough

You may write a Studio or viser plugin directly, using their own APIs. That code then
runs in that viewer only, and it is outside linker-sim's API: it may break when those
packages change.

## Not tested yet

- browsers on macOS and Linux ARM64
- computers with an integrated GPU (Intel or Apple); being measured before the 1.0 release
- an HTTPS reverse proxy
- several Studio spectators at once
- Studio's desktop window
