# HERO REDESIGN — Implementation TODO

## Step 1: Baseline review (done)
- Identified hero markup in `templates/index.html`.
- Identified hero styling in `static/css/style.css`.
- Noted potential CSS corruption risk (invalid placeholder fragments) and will avoid expanding that.

## Step 2: Markup hierarchy + accessibility (templates) (done)
- Add semantic grouping for CTAs and trust indicators within the hero.
- Ensure reading order: H1 → description → CTA group → trust → visual.
- Demote HUD/telemetry visual labels from screen reader focus where appropriate.

## Step 3: Typography + line length (CSS) (done)
- Strengthen `.hero-content h1` and `.hero-content p` hierarchy.
- Add max-width/line-height/letter-spacing tuning.
- Stabilize layout vs typewriter effect.

## Step 4: Spacing + visual hierarchy (CSS) (done)
- Improve vertical rhythm (badge/H1/description/CTA/trust).
- Ensure primary CTA has strongest visual weight.
- Adjust right visual container intensity.

## Step 5: Trust indicators cleanup (CSS/markup) (done)
- Ensure trust supports CTA and doesn’t compete.
- Make trust strip legible and static (non-jitter).

## Step 6: Responsive behavior (CSS) (done)
- Verify mobile stacking for CTA buttons.
- Ensure right visual scales without overlap.

## Step 7: Reduced motion + accessibility (CSS) (done)
- Respect `prefers-reduced-motion`.
- Add strong focus-visible for hero CTAs.

## Step 8: Validate (done)
- Smoke check `/` page rendering.
- Run project tests/lint/build if available.

