-- Seed test data for RAG retrieval testing
-- These are excerpts/paraphrases from public domain sources for development testing

-- Test chunk 1: Chopin legato technique (from Piano Mastery style)
INSERT OR IGNORE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, page_number, section_title, composers, pieces, techniques,
    source_hash
) VALUES (
    'test-001',
    'The secret of Chopin''s legato lies not in finger pressure alone, but in the singing quality of the melodic line. Each note must connect to the next as if drawn by an invisible thread. The wrist should remain supple, allowing the weight of the arm to transfer naturally from key to key.',
    '[Source: Piano Mastery by Harriette Brower, p.45]\n[Context: Chopin - Nocturnes - Legato, Touch]\n\nThe secret of Chopin''s legato lies not in finger pressure alone, but in the singing quality of the melodic line. Each note must connect to the next as if drawn by an invisible thread. The wrist should remain supple, allowing the weight of the arm to transfer naturally from key to key.',
    'book',
    'Piano Mastery',
    'Harriette Brower',
    'https://www.gutenberg.org/ebooks/26663',
    45,
    'Interview with Josef Hofmann',
    '["Chopin"]',
    '["Nocturne"]',
    '["legato", "touch", "singing tone"]',
    'a1b2c3d4e5f6test001'
);

-- Test chunk 2: Beethoven pedaling (from Czerny style)
INSERT OR IGNORE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, page_number, section_title, composers, pieces, techniques,
    source_hash
) VALUES (
    'test-002',
    'Beethoven insisted that the pedal should never obscure the harmonic clarity of a passage. He would say "the pedal is the soul of the piano" but only when used with discrimination. In his sonatas, the pedal markings are precise indications of his intentions, not mere suggestions.',
    '[Source: On the Proper Performance of All Beethoven''s Works for the Piano by Carl Czerny, p.23]\n[Context: Beethoven - Sonatas - Pedaling]\n\nBeethoven insisted that the pedal should never obscure the harmonic clarity of a passage. He would say "the pedal is the soul of the piano" but only when used with discrimination. In his sonatas, the pedal markings are precise indications of his intentions, not mere suggestions.',
    'book',
    'On the Proper Performance of All Beethoven''s Works for the Piano',
    'Carl Czerny',
    'https://imslp.org/wiki/Complete_Theoretical_and_Practical_Piano_Forte_School_(Czerny,_Carl)',
    23,
    'On the Use of the Pedal',
    '["Beethoven"]',
    '["Sonata"]',
    '["pedal", "pedaling", "clarity"]',
    'a1b2c3d4e5f6test002'
);

-- Test chunk 3: Bach articulation (from C.P.E. Bach Essay style)
INSERT OR IGNORE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, page_number, section_title, composers, pieces, techniques,
    source_hash
) VALUES (
    'test-003',
    'In fugal passages, each voice must maintain its own character through consistent articulation. The subject, when it enters, should be clearly distinguished from the accompanying voices. This is achieved not through dynamics alone, but through subtle differences in touch and timing.',
    '[Source: Essay on the True Art of Playing Keyboard Instruments by C.P.E. Bach, p.156]\n[Context: J.S. Bach - Fugue - Articulation, Voicing]\n\nIn fugal passages, each voice must maintain its own character through consistent articulation. The subject, when it enters, should be clearly distinguished from the accompanying voices. This is achieved not through dynamics alone, but through subtle differences in touch and timing.',
    'book',
    'Essay on the True Art of Playing Keyboard Instruments',
    'C.P.E. Bach',
    'https://imslp.org/wiki/Versuch_%C3%BCber_die_wahre_Art_das_Clavier_zu_spielen_(Bach,_Carl_Philipp_Emanuel)',
    156,
    'On Playing Fugues',
    '["Bach", "J.S. Bach"]',
    '["Fugue", "Well-Tempered Clavier"]',
    '["articulation", "voicing", "polyphony"]',
    'a1b2c3d4e5f6test003'
);

-- Test chunk 4: Liszt dynamics (masterclass style)
INSERT OR IGNORE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, timestamp_start, timestamp_end, speaker, composers, pieces, techniques,
    source_hash
) VALUES (
    'test-004',
    'The dramatic contrasts in Liszt require not just physical force but emotional commitment. When you play fortissimo, it must come from deep within the arm, through the shoulder. The pianissimo that follows should feel like a whisper, barely touching the keys yet projecting to the back of the hall.',
    '[Source: Masterclass on Liszt - Claudio Arrau (14:32)]\n[Context: Liszt - Sonata in B minor - Dynamics]\n\nThe dramatic contrasts in Liszt require not just physical force but emotional commitment. When you play fortissimo, it must come from deep within the arm, through the shoulder. The pianissimo that follows should feel like a whisper, barely touching the keys yet projecting to the back of the hall.',
    'masterclass',
    'Masterclass on Liszt Sonata',
    'Claudio Arrau',
    'https://youtube.com/watch?v=example123',
    872.0,
    885.0,
    'Claudio Arrau',
    '["Liszt"]',
    '["Sonata in B minor"]',
    '["dynamics", "fortissimo", "pianissimo", "projection"]',
    'a1b2c3d4e5f6test004'
);

-- Test chunk 5: Chopin rubato (from Piano Mastery style)
INSERT OR IGNORE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, page_number, section_title, composers, pieces, techniques,
    source_hash
) VALUES (
    'test-005',
    'Chopin''s rubato is often misunderstood. He himself compared it to a tree whose branches sway in the wind while the trunk remains firm. The left hand maintains a steady pulse like the trunk, while the right hand melody bends and stretches with expressive freedom. Too much rubato destroys the musical architecture.',
    '[Source: Piano Mastery by Harriette Brower, p.78]\n[Context: Chopin - Mazurkas - Rubato, Timing]\n\nChopin''s rubato is often misunderstood. He himself compared it to a tree whose branches sway in the wind while the trunk remains firm. The left hand maintains a steady pulse like the trunk, while the right hand melody bends and stretches with expressive freedom. Too much rubato destroys the musical architecture.',
    'book',
    'Piano Mastery',
    'Harriette Brower',
    'https://www.gutenberg.org/ebooks/26663',
    78,
    'Interview with Paderewski',
    '["Chopin"]',
    '["Mazurka"]',
    '["rubato", "timing", "rhythm", "expression"]',
    'a1b2c3d4e5f6test005'
);

-- Test chunk 6: Debussy tone color (masterclass style)
INSERT OR IGNORE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, timestamp_start, timestamp_end, speaker, composers, pieces, techniques,
    source_hash
) VALUES (
    'test-006',
    'In Debussy, the piano must become an orchestra of colors. Each register has its own timbre - the bass like cellos, the middle voice like clarinets, the treble like flutes. Use the different depths of touch to paint these colors. The half-pedal technique allows you to mix colors without muddying the texture.',
    '[Source: Debussy Interpretation Masterclass (08:15)]\n[Context: Debussy - Preludes - Timbre, Pedaling]\n\nIn Debussy, the piano must become an orchestra of colors. Each register has its own timbre - the bass like cellos, the middle voice like clarinets, the treble like flutes. Use the different depths of touch to paint these colors. The half-pedal technique allows you to mix colors without muddying the texture.',
    'masterclass',
    'Debussy Interpretation Masterclass',
    'Pascal Roge',
    'https://youtube.com/watch?v=example456',
    495.0,
    512.0,
    'Pascal Roge',
    '["Debussy"]',
    '["Preludes", "La Cathedrale Engloutie"]',
    '["timbre", "tone color", "pedal", "half-pedal", "voicing"]',
    'a1b2c3d4e5f6test006'
);

-- Test chunk 7: Mozart clarity (from historical source style)
INSERT OR IGNORE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, page_number, section_title, composers, pieces, techniques,
    source_hash
) VALUES (
    'test-007',
    'Mozart''s piano style demands absolute clarity and evenness. Every note in a scale passage must be equally weighted, like pearls on a string. The ornaments should sparkle without disturbing the phrase. Leopold Mozart wrote that his son''s playing was remarkable for its precision and the singing quality of slow movements.',
    '[Source: Leopold Mozart''s Letters, compiled by Emily Anderson, p.234]\n[Context: Mozart - Piano Sonatas - Clarity, Evenness]\n\nMozart''s piano style demands absolute clarity and evenness. Every note in a scale passage must be equally weighted, like pearls on a string. The ornaments should sparkle without disturbing the phrase. Leopold Mozart wrote that his son''s playing was remarkable for its precision and the singing quality of slow movements.',
    'letter',
    'Letters of Mozart and His Family',
    'Leopold Mozart',
    'https://archive.org/details/lettersofmozarta00mozauoft',
    234,
    'On Wolfgang''s Playing',
    '["Mozart"]',
    '["Piano Sonata"]',
    '["clarity", "evenness", "ornaments", "scales"]',
    'a1b2c3d4e5f6test007'
);

-- Test chunk 8: Rachmaninoff tone production
INSERT OR IGNORE INTO pedagogy_chunks (
    chunk_id, text, text_with_context, source_type, source_title, source_author,
    source_url, page_number, section_title, composers, pieces, techniques,
    source_hash
) VALUES (
    'test-008',
    'The Rachmaninoff sound requires a particular approach to tone production. The key velocity must be controlled precisely - attack the key with speed but catch the depth. His cantabile passages should ring out above the accompaniment like a great singer. Practice the melody alone until it truly sings.',
    '[Source: Piano Mastery by Harriette Brower, p.112]\n[Context: Rachmaninoff - Concertos - Tone, Singing]\n\nThe Rachmaninoff sound requires a particular approach to tone production. The key velocity must be controlled precisely - attack the key with speed but catch the depth. His cantabile passages should ring out above the accompaniment like a great singer. Practice the melody alone until it truly sings.',
    'book',
    'Piano Mastery',
    'Harriette Brower',
    'https://www.gutenberg.org/ebooks/26663',
    112,
    'Interview with Josef Lhevinne',
    '["Rachmaninoff"]',
    '["Piano Concerto"]',
    '["tone", "singing tone", "cantabile", "voicing"]',
    'a1b2c3d4e5f6test008'
);
