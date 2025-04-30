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

def process_file(input_file, output_file, is_fake):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        outfile.write('id,dataset_id,title,content,classification\n')
        
        csv_reader = csv.reader(infile)
        next(csv_reader)  
        
        for row in csv_reader:
            if len(row) >= 3:  
                news_id = row[0]
                title = row[2].replace('"', '""')  
                content = ""
                classification = "fake" if is_fake else "real"
                
                outfile.write(f'{news_id},0,"{title}","{content}",{classification}\n')

def combine_formatted_files(output_dir, final_output):
    with open(final_output, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write('id,dataset_id,title,content,classification\n')
        
        for filename in os.listdir(output_dir):
            if filename.startswith('formatted_') and filename.endswith('.csv'):
                input_path = os.path.join(output_dir, filename)
                print(f"Combining {filename}...")
                
                with open(input_path, 'r', encoding='utf-8') as infile:
                    next(infile)  
                    for line in infile:
                        outfile.write(line)

def main():
    base_dir = '.'  
    dataset_dir = os.path.join(base_dir, 'dataset')
    output_dir = os.path.join(base_dir, 'formated')
    
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_process = [
        ('gossipcop_fake.csv', True),
        ('gossipcop_real.csv', False),
        ('politifact_fake.csv', True),
        ('politifact_real.csv', False)
    ]
    
    for filename, is_fake in files_to_process:
        input_path = os.path.join(dataset_dir, filename)
        output_path = os.path.join(output_dir, f'formatted_{filename}')
        print(f"Processing {filename}...")
        process_file(input_path, output_path, is_fake)
        print(f"Completed processing {filename}")
    
    final_output = os.path.join(base_dir, 'fakenewsnet-formated.csv')
    print("\nCombining all formatted files...")
    combine_formatted_files(output_dir, final_output)
    print("Created combined file: fakenewsnet-formated.csv")

if __name__ == '__main__':
    main() 