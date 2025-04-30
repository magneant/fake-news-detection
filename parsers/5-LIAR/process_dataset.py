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

def convert_label_to_binary(label):
    real_labels = ['true', 'mostly-true']
    return "real" if label.lower() in real_labels else "fake"

def process_tsv_file(input_file, output_writer):
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) >= 3:  
                news_id = parts[0]
                label = parts[1]
                statement = parts[2].replace('"', '""')  
                
                content = ""
                
                classification = convert_label_to_binary(label)
                
                output_writer.write(f'{news_id},4,"{statement}","{content}",{classification}\n')

def main():
    base_dir = '.'  
    output_dir = os.path.join(base_dir, 'formated')
    
    os.makedirs(output_dir, exist_ok=True)
    
    input_files = [
        os.path.join(base_dir, 'train.tsv'),
        os.path.join(base_dir, 'test.tsv'),
        os.path.join(base_dir, 'valid.tsv')
    ]
    
    output_file = os.path.join(output_dir, 'formatted_LIAR.csv')
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        # Write header
        outfile.write('id,dataset_id,title,content,classification\n')
        
        for input_file in input_files:
            file_name = os.path.basename(input_file)
            print(f"Processing {file_name}...")
            process_tsv_file(input_file, outfile)
            print(f"Completed processing {file_name}")
    
    final_output = os.path.join(base_dir, 'liar-formated.csv')
    print("\nCreating final formatted file...")
    with open(output_file, 'r', encoding='utf-8') as infile, \
         open(final_output, 'w', encoding='utf-8') as outfile:
        outfile.write(infile.read())
    print("Created final file: liar-formated.csv")

if __name__ == '__main__':
    main() 