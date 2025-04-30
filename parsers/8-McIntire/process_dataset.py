import csv
import os
import sys

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def clean_text(text):
    """Clean text by removing extra whitespace and newlines"""
    if text:
        return ' '.join(text.replace('\n', ' ').split())
    return text

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        outfile.write('id,dataset_id,title,content,classification\n')
        
        csv_reader = csv.DictReader(infile)
        
        counter = 1
        for row in csv_reader:
            if row.get('title') and row.get('label'):  
                news_id = f"mcintire-{counter}"
                counter += 1
                
                title = clean_text(row['title'])
                if title:  
                    title = title.replace('"', '""') 
                    
                    content = ""
                    if 'text' in row and row['text']:
                        content = clean_text(row['text'])
                        content = content.replace('"', '""') 
                    
                    classification = row['label'].lower()
                    
                    outfile.write(f'{news_id},7,"{title}","{content}",{classification}\n')

def main():
    base_dir = '.'  
    input_file = os.path.join(base_dir, 'Fake and Real News Dataset.csv')
    output_dir = os.path.join(base_dir, 'formated')
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'formatted_mcintire.csv')
    print("Processing Fake and Real News Dataset.csv...")
    process_file(input_file, output_file)
    print("Completed processing Fake and Real News Dataset.csv")
    
    final_output = os.path.join(base_dir, 'mcintire-formated.csv')
    print("\nCreating final formatted file...")
    with open(output_file, 'r', encoding='utf-8') as infile, \
         open(final_output, 'w', encoding='utf-8') as outfile:
        outfile.write(infile.read())
    print("Created final file: mcintire-formated.csv")

if __name__ == '__main__':
    main() 