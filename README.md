# Delayed Reward

A plain-language blog series on the textbook *Reinforcement Learning: Foundations* (Shie Mannor, Yishay Mansour, Aviv Tamar).

**Read it:** https://kookie14.github.io/Reinforcement-Learning-Foundations/

## Layout

```text
blog/            posts as Markdown (NN-slug.md) + series.json (learning path, site settings)
blog/code/       companion scripts linked from posts
notes/           reading notes and the book map
site/            generator: build.py, templates/, assets/
.github/         GitHub Actions workflow that builds and deploys the site
```

## Write a post

1. Create `blog/NN-slug.md` starting with front matter:

   ```yaml
   ---
   number: "02"
   title: States, Actions, Rewards and Policies
   part: 1
   date: 2026-09-24
   chapters: Ch. 3.1–3.2, 5.1
   summary: One sentence shown on the home page and in link previews.
   ---
   ```

2. Write in Markdown. Math uses `$...$` and `$$...$$`, diagrams use ```` ```mermaid ```` blocks, and a dollar sign in prose is `\$`.
3. Preview locally:

   ```bash
   pip install -r site/requirements.txt
   python3 site/build.py --serve     # http://localhost:8000
   ```

4. Commit and push to `main`. GitHub Actions rebuilds and deploys the site.

## One-time GitHub setup

In the repository: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
