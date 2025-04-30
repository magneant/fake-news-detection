import csv
import os
import sys
import hashlib

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def generate_id(title, text):
    combined = f"{title}{text[:100]}".encode('utf-8')
    return hashlib.md5(combined).hexdigest()[:15]

def sanitize_content(text: str) -> str:
    no_breaks = text.replace('\r', ' ').replace('\n', ' ')
    return ' '.join(no_breaks.split())

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.writer(
            outfile,
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            delimiter=',',
            escapechar='\\',
            doublequote=True
        )
        
        writer.writerow(['id', 'dataset_id', 'title', 'content', 'classification'])
        
        rows_processed = 0
        for row in reader:
            title = row.get('title', '').strip()
            label = row.get('label', '').strip()
            text  = row.get('text', '').strip()
            
            if not title or not label:
                continue
            
            content = sanitize_content(text)
            
            news_id = generate_id(title, content)
            classification = "real" if label == '1' else "fake"
            
            writer.writerow([
                news_id,
                '2',      
                title,
                content,
                classification
            ])
            
            rows_processed += 1
            if rows_processed % 1000 == 0:
                print(f"Processed {rows_processed} rows...")

def main():
    base_dir    = '.'
    input_file  = os.path.join(base_dir, 'WELFake_Dataset.csv')
    output_dir  = os.path.join(base_dir, 'formated')
    os.makedirs(output_dir, exist_ok=True)
    
    temp_file   = os.path.join(output_dir, 'formatted_WELFake.csv')
    final_file  = os.path.join(base_dir, 'welfake-formated.csv')
    
    print("Processing WELFake_Dataset.csv...")
    process_file(input_file, temp_file)
    print("Completed processing WELFake_Dataset.csv")
    
    os.replace(temp_file, final_file)
    print(f"Created final file: {final_file}")

if __name__ == '__main__':
    main()
