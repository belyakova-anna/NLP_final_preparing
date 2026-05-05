# NLP — Final Exam Tickets · Reader

Editorial-style static reader for the 20 final exam tickets (Innopolis, S26).

## Local preview

```bash
cd site
python3 -m http.server 8080
# open http://localhost:8080
```

## Stack

- Vanilla HTML / CSS / JS — no build step
- [marked](https://marked.js.org) — Markdown rendering
- [KaTeX](https://katex.org) — math typesetting
- Fonts: Instrument Serif, IBM Plex Sans, JetBrains Mono

## Deployment

Pushed to `main`, the workflow at `.github/workflows/deploy.yml` will:

1. Use ticket markdown directly from `site/tickets/ticket-NN.md`.
2. Upload the `site/` folder as a Pages artifact.
3. Deploy via `actions/deploy-pages`.

### One-time setup

In your GitHub repository:

1. **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `main` (or run the workflow manually from the Actions tab).
3. The deployed URL appears in the workflow output and at `Settings → Pages`.

## Keyboard shortcuts

- `←` / `→` — previous / next ticket
- `G` — back to index
- `/` — focus search
