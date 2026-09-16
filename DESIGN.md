# 2048 AI design direction

## 1. Visual theme and atmosphere

The product is tactile, warm, and deliberate. It should feel like moving numbered ceramic tiles across a paper game board, with quiet surfaces in manual play and one unmistakable signal when AI takes control: a restrained amber pulse that connects the chosen direction to the resulting move.

The board is the visual anchor. Supporting controls stay cardless where possible and use alignment, type, and background steps instead of a dashboard grid.

## 2. Color palette and roles

All authored colors use OKLCH.

| Token | Value | Role |
| --- | --- | --- |
| `canvas` | `oklch(0.955 0.024 80)` | warm paper page |
| `surface` | `oklch(0.985 0.012 82)` | controls and score surfaces |
| `surface-muted` | `oklch(0.91 0.025 72)` | subdued information |
| `ink` | `oklch(0.28 0.035 50)` | primary text |
| `ink-muted` | `oklch(0.51 0.035 64)` | explanatory text |
| `board` | `oklch(0.62 0.05 60)` | game board body |
| `slot` | `oklch(0.77 0.03 70)` | empty cells |
| `amber` | `oklch(0.72 0.15 68)` | AI state and primary action |
| `amber-deep` | `oklch(0.49 0.12 55)` | amber text and pressed state |
| `clay` | `oklch(0.58 0.16 30)` | errors and destructive confirmation |
| `olive` | `oklch(0.56 0.10 115)` | ready and healthy states |

Tile colors progress from warm ivory through sand, apricot, clay, amber, and deep brown. Text on a colored tile uses a darker or lighter shade of the same warm hue, never neutral gray.

## 3. Typography rules

Brand words: tactile, warm, deliberate. Reflex choices rejected: Inter, DM Sans, and Space Grotesk. The display and numeral face is `Avenir Next`, chosen for its humanist rhythm and stable tabular-feeling numerals; Chinese text uses `PingFang SC`, then the system sans fallback.

- Product title: 42–64 px, weight 750, line-height 0.94, letter-spacing `-0.022em`.
- Score numerals: 24–32 px, weight 700, tabular numerals, letter-spacing `-0.012em`.
- Tile numerals: responsive 24–52 px, weight 750, tabular numerals.
- Body and controls: 14–16 px, weight 500–650, line-height 1.45.
- Small status labels: 12–13 px, weight 700, letter-spacing `0.055em` only when rendered uppercase Latin; Chinese labels remain at normal tracking.

Headings use balanced wrapping. Body copy uses pretty wrapping. Font smoothing is set once on the root.

## 4. Component styling

- Primary button: amber fill, `14px` radius, dark warm text, minimum 44 px height, subtle two-layer shadow. Hover lifts by 1 px on pointer devices; press scales to 0.96.
- Secondary button: warm surface fill with no hard border, same geometry, visible focus ring in amber-deep.
- Destructive confirmation: clay action with plain-language copy. It is the only modal because data loss requires focus lock.
- Speed selector: segmented control on a single muted surface, one amber inset selection, 44 px targets.
- Score blocks: compact aligned counters, not generic cards. Labels sit above tabular numerals.
- AI status: a small state dot, plain status copy, backend label, and last-direction glyph. No confidence percentage.
- Board: deeply rounded `22px` outer geometry, `14px` cell radius, dimensional shadow, no decorative border.

Component states must cover default, hover where pointer-capable, focus-visible, active, and disabled. Disabled controls lose chroma but retain readable contrast.

## 5. Layout principles

The desktop first viewport uses an asymmetric two-column composition: identity and scores form a narrow rail; the board owns the larger column; controls align directly to the board rather than forming a third card. At compact widths, content becomes a single column with the title and scores sharing the first row.

Spacing follows `4, 8, 12, 16, 24, 32, 48, 64`. The board is square, fluid, and capped near 560 px. The page supports 320 px width and applies safe-area padding.

Radius scale: `sm 8px`, `md 14px`, `lg 22px`, `pill 999px`. Nested radii remain concentric.

## 6. Depth and elevation

- Canvas: flat warm paper.
- Controls: surface color step plus `0 1px 3px rgb(70 45 25 / 0.12)`.
- Board: `0 18px 45px rgb(70 45 25 / 0.20), 0 3px 9px rgb(70 45 25 / 0.14)`.
- Raised tile during motion: scale and opacity only; no animated shadow.
- Modal: solid surface and stronger shadow, never glass or backdrop blur.

## 7. Do and do not

- Do let the board dominate the first screen.
- Do keep manual play quiet and make amber meaningful only for AI/action state.
- Do use numerals as the non-color cue for every tile.
- Do write direct Chinese status and error messages.
- Do animate only transform and opacity, using `cubic-bezier(0.16, 1, 0.3, 1)`.
- Do not use blue/purple gradients, glassmorphism, or a generic dashboard card grid.
- Do not use `transition: all`, elastic bounce, or layout-property animation.
- Do not hide model failures behind random moves or decorative loading loops.
- Do not let AI status compete with the score or board.
- Do not add imagery: PixiJS geometry and type are the product visual.

## 8. Responsive behavior

- `>= 900px`: asymmetric two-column layout, board at 480–560 px.
- `600–899px`: board-centered single column, title and scores in one compact header.
- `< 600px`: edge-aware single column, board fills available width, controls wrap in reading order.
- Minimum viewport: 320 CSS px.
- All targets are at least 40×40 px, primary controls 44 px high.
- Hover styles are guarded by pointer capability.
- Reduced-motion mode removes AI pulse and makes tile transitions effectively immediate.

## 9. Agent prompt guide

Quick tokens: canvas `oklch(0.955 0.024 80)`, surface `oklch(0.985 0.012 82)`, ink `oklch(0.28 0.035 50)`, board `oklch(0.62 0.05 60)`, amber `oklch(0.72 0.15 68)`, radii `8/14/22px`.

- Create the game header on the paper canvas, title at 52 px weight 750 line-height 0.94 letter-spacing -0.022em, with compact score counters using tabular numerals.
- Create the PixiJS board as a square warm-brown surface with 22 px outer radius, 14 px tile radius, and 12 px visual gaps; use no image assets.
- Create an AI control strip with a 44 px amber primary button, segmented speed selector, state dot, backend label, and last direction; amber appears only for active AI state.
- Create a destructive restart confirmation on the solid warm surface, 22 px radius, clay action, secondary cancel action, and no backdrop blur.
- Create the compact 375 px layout with safe-area padding, full-width board, 44 px controls, and no horizontal overflow.

## CSS strategy

Use CSS Modules only. Define global reset, document tokens, and accessibility utilities through `:global` selectors inside the root module; component styling remains locally scoped. Do not introduce Tailwind, CSS-in-JS, or a second styling system.
