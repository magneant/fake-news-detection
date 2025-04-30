import csv
import os
import sys

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        outfile.write('id,dataset_id,title,content,classification\n')
        
        csv.register_dialect(
            'recovery',
            delimiter=',',
            quotechar='"',
            doublequote=True,
            skipinitialspace=True,
            quoting=csv.QUOTE_MINIMAL
        )
        reader = csv.DictReader(infile, dialect='recovery')
        
        for row in reader:
            title = row.get('title')
            rel   = row.get('reliability')
            if not title or rel is None:
                continue

            news_id   = row['news_id']
            title_esc = title.replace('"', '""')

            raw = row.get('body_text', '')
            no_lines = raw.replace('\r', ' ').replace('\n', ' ')
            collapsed = ' '.join(no_lines.split())
            content = collapsed.replace('"', '""')

            classification = "real" if rel.strip() == '1' else "fake"

            outfile.write(
                f'{news_id},6,"{title_esc}","{content}",{classification}\n'
            )

def main():
    base_dir    = '.'
    input_file  = os.path.join(base_dir, 'Recovery News Data.csv')
    output_dir  = os.path.join(base_dir, 'formated')
    os.makedirs(output_dir, exist_ok=True)
    
    formatted = os.path.join(output_dir, 'formatted_recovery.csv')
    print("Processing Recovery News Data.csv...")
    process_file(input_file, formatted)
    print("Completed processing Recovery News Data.csv")
    
    final_output = os.path.join(base_dir, 'recovery-formated.csv')
    with open(formatted, 'r', encoding='utf-8') as src, \
         open(final_output, 'w', encoding='utf-8') as dst:
        dst.write(src.read())
    print("Created final file: recovery-formated.csv")

if __name__ == '__main__':
    main()
