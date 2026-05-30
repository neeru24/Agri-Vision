# TODO - Unified Form Input System (Agri-Vision)

## Step 1: CSS foundation
- [x] Add design tokens for input system (radius, heights, padding, focus ring, validation colors) to `static/css/style.css`

- [ ] Add base styles for `input/select/textarea` with `:hover` + `:focus-visible`
- [ ] Add error/success hooks via `[aria-invalid="true"]`, `.is-invalid`, `.is-valid`
- [ ] Add helper classes: `.form-field`, `.form-label`, `.form-hint`, `.form-error`
- [ ] Add reduced-motion handling for input transitions

## Step 2: Auth forms
- [ ] Modernize inputs in `templates/login.html`
- [ ] Modernize inputs in `templates/register.html`

## Step 3: Upload + prediction entry form
- [ ] Apply unified classes to weather city input + controls in `templates/upload.html`
- [ ] Apply unified classes / styling hooks to file upload area and hidden upload inputs

## Step 4: Filters/search across app
- [ ] Update remaining templates with inputs/selects/search to use unified system

## Step 5: Validation + A11y
- [ ] Ensure aria-invalid usage where applicable (without breaking backend)
- [ ] Ensure labels remain correctly associated (for/id)

## Step 6: Quality gates
- [ ] Run lint (if available)
- [ ] Run typecheck (if available)
- [ ] Run tests (`pytest`)
- [ ] Run production build checks (if available)

