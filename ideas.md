# HH RAG Frontend — Design Direction

## Three stylistic approaches

### Theme Name: Signal Room
Very dark, editorial, and instrument-like: a focused knowledge interface with a warm signal accent and high legibility.
Probability: 0.07

### Theme Name: Monsoon Index
A moody coastal palette with mineral blues, wet-surface texture, and restrained data notation.
Probability: 0.03

### Theme Name: Paper Cutover
A light, tactile research notebook aesthetic with ink typography, offset labels, and a single saturated red accent.
Probability: 0.09

## Selected approach: Signal Room

### Design Movement
Contemporary editorial systems design: the visual language of a broadcast control room translated into a calm, premium research instrument.

### Core Principles
1. **Answer-first hierarchy:** the returned answer is the largest and most composed object on the page.
2. **Instrumental clarity:** statuses, latency, language, and request metadata behave like concise readouts, never decorative dashboards.
3. **Warm signal on mineral dark:** an ownable saffron-orange accent cuts through blue-black surfaces to indicate action, voice, and grounded confidence.
4. **Asymmetry with restraint:** the layout uses a narrow rail and a wide answer field rather than a centered generic card stack.

### Color Philosophy
The base is a blue-black near-black that feels like a quiet studio after hours. Soft slate surfaces create depth without glossy gradients. Saffron orange is reserved for the active signal: microphone, submit action, live recording, and grounded confirmation. A muted sea-glass green communicates successful grounding without pretending to be a diagnostic dashboard.

### Layout Paradigm
A two-column editorial workspace: a compact left rail introduces the product and interaction controls, while the right side is a generous answer canvas. On mobile, the rail becomes a top band and the answer canvas follows naturally below. Empty state composition keeps the interaction anchored left and the result zone visibly open.

### Signature Elements
- A vertical **signal rail** with a saffron live marker and compact system readouts.
- Fine-grain **index labels** in uppercase mono type for stages, latency, and request identity.
- A small concentric **voice aperture** motif around the microphone control, animated only while recording.

### Interaction Philosophy
Interactions should feel like operating a precise instrument: immediate press feedback, clear state transitions, no theatrical loading deception. The microphone is a primary action, but every state is named honestly and the user can always recover by switching to text.

### Animation
Use short ease-out transitions under 240ms for controls and disclosure panels. Recording uses a restrained radial pulse and a timer; processing uses a moving saffron hairline rather than fake streaming. Answer reveal uses opacity plus a small upward transform. Respect reduced-motion preferences and keep keyboard submission instant.

### Typography System
Display: **DM Sans** with 600–700 weight for headings and answer lead-ins. Body: **Source Sans 3** for readable explanatory copy. Metadata: **IBM Plex Mono** in uppercase with generous tracking. Headlines use tight line-height; answer copy uses relaxed 1.65 line-height.

### Brand Essence
**HH RAG is a voice-first knowledge instrument for fast, grounded answers from multilingual retrieval — built for demos where trust has to be visible.** Personality: precise, warm, assured.

### Brand Voice
Headlines are direct and composed. CTAs are verbs with a clear outcome. Microcopy states what is happening without overpromising backend streaming or telemetry.

Example lines:
- “Ask once. See what the knowledge can support.”
- “Listening locally. Grounding remotely.”

### Wordmark & Logo
The mark is a bold, text-free aperture: two offset saffron brackets forming an abstract **H** around a small central dot, suggesting both a microphone opening and a retrieval target. The wordmark pairs a compact uppercase `HH` with a thin monospaced `RAG` label rather than a default brand font treatment.

### Signature Brand Color
**Signal Saffron — #F6A34A**, used sparingly for action, voice, and grounded attention.

## Style Decisions

- The Signal Room surface should feel like a calm broadcast-control desk, not distressed sci-fi: the primary app field is now clean and mineral, with texture reserved for supporting visual assets.
- The HH RAG lockup uses a custom-feeling contrast between compact uppercase HH and thin monospaced RAG, paired with the saffron aperture mark.
- Empty states reinforce answer-first hierarchy by treating the right-side answer canvas as the main knowledge artifact, while input controls remain compact and operational.
