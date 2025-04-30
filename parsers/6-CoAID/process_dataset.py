import csv
import os
import sys
import hashlib
import glob
from collections import Counter

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

BOILERPLATE = [
    "this website is using a security service",
    "to protect itself from online attacks",
]

def generate_id(title, url):
    combined = f"{title}{url}".encode('utf-8')
    return hashlib.md5(combined).hexdigest()[:15]

def sanitize_field(text: str) -> str:
    clean = text.replace('"', '""').replace('\r', ' ').replace('\n', ' ')
    return ' '.join(clean.split())

def looks_boilerplate(content: str) -> bool:
    low = content.lower()
    return any(phrase in low for phrase in BOILERPLATE)

def has_over_repeated_sentence(content: str, threshold: int = 3) -> bool:
    """
    Split content into sentences on '. ' then count occurrences.
    If any sentence appears more than `threshold` times, return True.
    """
    sentences = [s.strip() for s in content.split('. ') if s.strip()]
    counts = Counter(sentences)
    return any(count > threshold for count in counts.values())

def process_claim_file(input_file, output_writer, is_fake, stats):
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            title = row.get('title', '').strip()
            if not title:
                continue

            raw = row.get('claim', '')
            content = sanitize_field(raw)

            # skip boilerplate or over-repetition
            if looks_boilerplate(content) or has_over_repeated_sentence(content):
                stats['skipped'] += 1
                continue

            news_id       = generate_id(title, row.get('news_url', ''))
            classification = "fake" if is_fake else "real"
            title_san      = sanitize_field(title)

            output_writer.write(
                f'{news_id},5,"{title_san}","{content}",{classification}\n'
            )
            stats['written'] += 1

def process_news_file(input_file, output_writer, is_fake, stats):
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            raw_title = row.get('title') or row.get('newstitle') or ''
            title = raw_title.strip()
            if not title:
                continue

            content_raw  = row.get('content', '')
            abstract_raw = row.get('abstract', '')
            combined     = f"{content_raw} {abstract_raw}".strip()
            content      = sanitize_field(combined)

            if looks_boilerplate(content) or has_over_repeated_sentence(content):
                stats['skipped'] += 1
                continue

            news_id       = generate_id(title, row.get('news_url', ''))
            classification = "fake" if is_fake else "real"
            title_san      = sanitize_field(title)

            output_writer.write(
                f'{news_id},5,"{title_san}","{content}",{classification}\n'
            )
            stats['written'] += 1

def process_directory(month_dir, output_base_dir):
    month_name = os.path.basename(month_dir)
    output_dir = os.path.join(output_base_dir, month_name)
    os.makedirs(output_dir, exist_ok=True)

    patterns = {
        'ClaimFake': ('Claim*Fake*.csv', True),
        'ClaimReal': ('Claim*Real*.csv', False),
        'NewsFake':  ('News*Fake*.csv', True),
        'NewsReal':  ('News*Real*.csv', False),
    }

    for file_type, (pattern, is_fake) in patterns.items():
        stats = {'written': 0, 'skipped': 0}
        out_path = os.path.join(output_dir, f'formatted_{file_type}.csv')
        with open(out_path, 'w', encoding='utf-8', newline='') as outfile:
            outfile.write('id,dataset_id,title,content,classification\n')
            for input_file in glob.glob(os.path.join(month_dir, pattern)):
                print(f"Processing {os.path.basename(input_file)}…")
                if file_type.startswith('Claim'):
                    process_claim_file(input_file, outfile, is_fake, stats)
                else:
                    process_news_file(input_file, outfile, is_fake, stats)
            print(f" → {file_type}: written={stats['written']}  skipped={stats['skipped']}")

def combine_all_files(base_dir, output_file):
    print("\nCombining all formatted files…")
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write('id,dataset_id,title,content,classification\n')
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if fname.startswith('formatted_') and fname.endswith('.csv'):
                    path = os.path.join(root, fname)
                    with open(path, 'r', encoding='utf-8') as infile:
                        next(infile)  # skip header
                        for line in infile:
                            outfile.write(line)

def main():
    base_dir    = '.'
    coaid_dir   = os.path.join(base_dir, 'CoAID-master')
    output_dir  = os.path.join(base_dir, 'formated')
    os.makedirs(output_dir, exist_ok=True)

    for month_dir in glob.glob(os.path.join(coaid_dir, '*-*-*')):
        if os.path.isdir(month_dir):
            process_directory(month_dir, output_dir)

    final_csv = os.path.join(base_dir, 'coaid-formated.csv')
    combine_all_files(output_dir, final_csv)
    print(f"\nAll done — combined into {final_csv}")

if __name__ == '__main__':
    main()
