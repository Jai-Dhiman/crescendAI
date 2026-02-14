# Research Agent Prompt: Find All Downloadable Masterclass Videos

## Objective

Find every publicly accessible video of a **piano masterclass** on **Chopin Ballade No. 1 in G minor, Op. 23** where a teacher works with a student in a stop-and-correct format. Compile results into structured JSON.

## What Qualifies

A video qualifies if ALL of the following are true:

1. **Piece**: The student is playing Chopin Ballade No. 1 in G minor, Op. 23 (full or partial)
2. **Format**: A teacher listens to a student play, then stops/interrupts to give feedback. The student may replay the passage. This is the "masterclass" or "coaching" format.
3. **Downloadable**: The video is on a platform where the audio can be extracted (YouTube, Vimeo, Bilibili, Dailymotion, or any yt-dlp supported site)
4. **Audio quality**: The piano playing is clearly audible (not just a phone recording of a screen)
5. **Language**: Any language for the teacher's speech is fine (we have transcription). The playing is what matters.

## What Does NOT Qualify

Exclude these:
- **Solo performances** (no teacher feedback)
- **Lecture-only** videos where a teacher talks about the piece without a student present
- **Reaction videos** or commentary over a recording
- **Previews/trailers** under 3 minutes
- **Duplicate uploads** of the same masterclass on different channels (keep the highest quality version)
- **Audio-only podcasts** discussing the piece

## Search Strategy

Search each of these query patterns across YouTube. For each query, examine at least the first 30 results:

### Primary queries (English)
- `Chopin Ballade No 1 masterclass`
- `Chopin Ballade 1 G minor masterclass`
- `Chopin Ballade Op 23 masterclass`
- `Chopin Ballade No 1 master class piano`
- `Chopin Ballade 1 piano lesson`
- `Chopin Ballade No 1 coaching session`
- `Chopin Ballade 1 piano class`

### Teacher-specific queries
For each of these known masterclass teachers, search `[teacher name] Chopin Ballade`:
- Lang Lang
- Daniel Barenboim
- Murray Perahia
- Garrick Ohlsson
- Mitsuko Uchida
- Andras Schiff
- Krystian Zimerman
- Leon Fleisher
- Seymour Bernstein
- Richard Goode
- Yuja Wang
- Boris Berman
- Jerome Rose
- Dmitri Bashkirov
- Arie Vardi
- Joaquin Achucarro
- Robert Levin
- Emanuel Ax
- Menahem Pressler
- Stephen Hough
- Marc-Andre Hamelin
- Dang Thai Son
- Kevin Kenner
- Piotr Paleczny
- Nelson Freire
- Stanislav Ioudenitch

### Institution-specific queries
Search `Chopin Ballade masterclass` on these channels/sites:
- Juilliard School (YouTube channel)
- Royal College of Music (YouTube channel)
- tonebase Piano (YouTube channel)
- Verbier Festival (YouTube channel)
- Pianist Magazine (YouTube channel)
- Jerusalem Music Centre (YouTube channel)
- Guildhall School (YouTube channel)
- Saline royale Academy (YouTube channel)
- Mozarteum Salzburg
- Hochschule fur Musik Hanns Eisler
- Lake District Summer Music (YouTube channel)
- Buchmann-Mehta School of Music (YouTube channel)
- Music Academy of the West (YouTube channel)
- Fryderyk Chopin Institute / chopin.nifc.pl
- Chopin Competition masterclasses
- International Piano Academy Lake Como

### Non-English queries
- `Chopin Ballade Nr 1 Meisterkurs` (German)
- `Chopin Ballade 1 classe de maitre` (French)
- `Chopin Ballade 1 clase magistral piano` (Spanish)
- `Chopin Ballade 1 masterclass pianoforte` (Italian)
- `ショパン バラード1番 マスタークラス` (Japanese)
- `쇼팽 발라드 1번 마스터클래스` (Korean)
- `肖邦第一叙事曲 大师课` (Chinese)
- `Шопен Баллада 1 мастер-класс` (Russian)
- `Chopin Ballada nr 1 kurs mistrzowski` (Polish)

### Competition masterclass queries
- `Chopin Competition masterclass Ballade`
- `International Chopin Piano Competition masterclass`
- `Chopin Competition 2021 masterclass`
- `Chopin Competition 2015 masterclass`
- `Cliburn competition masterclass Chopin Ballade`
- `Leeds piano competition masterclass Chopin`
- `Rubinstein competition masterclass Chopin`

### Platform-specific searches
- **Bilibili**: Search `肖邦第一叙事曲 大师课` and `Chopin Ballade masterclass`
- **Vimeo**: Search `Chopin Ballade masterclass`
- **Dailymotion**: Search `Chopin Ballade masterclass`
- **Medici.tv**: Check if any Chopin Ballade masterclasses are listed (note if behind paywall)

## Output Format

Return a JSON array. Each entry must have these fields:

```json
[
  {
    "url": "https://www.youtube.com/watch?v=XXXXX",
    "video_id": "XXXXX",
    "platform": "youtube",
    "title": "Exact video title as shown on the platform",
    "channel": "Channel or uploader name",
    "teacher": "Name of the masterclass teacher, or null if unknown",
    "student": "Name of the student, or null if unknown",
    "duration_seconds": 1234,
    "upload_date": "YYYY-MM-DD or null",
    "language": "en",
    "description_snippet": "First 200 chars of video description",
    "format_confidence": "high|medium|low",
    "format_notes": "Brief note on why this qualifies or any concerns about format",
    "piece_coverage": "full|partial|excerpt|unknown",
    "piece_notes": "e.g. 'Covers exposition only' or 'Full ballade plus discussion' or 'Multiple pieces, Ballade starts at 12:30'",
    "timestamp_offset": null,
    "audio_quality_estimate": "high|medium|low",
    "duplicate_of": null
  }
]
```

### Field Details

- **format_confidence**:
  - `high`: Title/description explicitly says masterclass AND you can confirm student+teacher format
  - `medium`: Appears to be a masterclass from title/thumbnail but format not 100% confirmed
  - `low`: Might be a masterclass, needs manual verification
- **piece_coverage**: Whether the Ballade No. 1 is the full focus or just part of a multi-piece session
- **timestamp_offset**: If the Ballade section starts partway through a longer video, note the approximate start time in seconds (e.g. `750` for 12:30). Null if it starts from the beginning.
- **duplicate_of**: If this appears to be the same masterclass uploaded elsewhere, reference the url of the primary version

## Deduplication Rules

If the same masterclass appears on multiple channels:
1. Keep the version from the official institution/teacher channel as primary
2. Mark other uploads with `duplicate_of` pointing to the primary
3. Include duplicates in the output (we may need them as fallback if primary gets deleted)

## Final Checklist

Before returning results, verify:
- [ ] Every entry has a valid, complete URL
- [ ] No performance-only videos slipped through
- [ ] No lecture-only videos slipped through
- [ ] Teacher names are consistently spelled across entries
- [ ] Duration is in seconds (not minutes)
- [ ] Duplicates are properly cross-referenced
- [ ] Results from non-English searches are included
- [ ] format_confidence accurately reflects certainty level

## Expected Yield

Based on the popularity of this piece in masterclass settings, expect to find approximately 15-40 qualifying videos. If you find fewer than 10, revisit the search queries and broaden. If you find more than 50, tighten the format_confidence filter.
