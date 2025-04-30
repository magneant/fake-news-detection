import csv
import os
import sys
import xml.etree.ElementTree as ET
from glob import glob

def process_xml_file(xml_file):
    """Process a single XML file and return the needed information"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        title = root.find('title').text if root.find('title') is not None else ''
        veracity = root.find('veracity').text if root.find('veracity') is not None else ''
        
        text = root.find('mainText').text if root.find('mainText') is not None else ''
        
        classification = "real" if veracity == "mostly true" else "fake"
        
        return title, text, classification
    except Exception as e:
        print(f"Error processing {xml_file}: {str(e)}")
        return None, None, None

def process_files(input_dir, output_file):
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write('id,dataset_id,title,content,classification\n')
        
        counter = 1
        for xml_file in glob(os.path.join(input_dir, '*.xml')):
            print(f"Processing file {counter}...")
            
            title, content, classification = process_xml_file(xml_file)
            if title and classification:
                title = ' '.join(title.replace('\n', ' ').split())
                title = title.replace('"', '""') 
                
                if content:
                    content = ' '.join(content.replace('\n', ' ').split())
                    content = content.replace('"', '""') 
                else:
                    content = ""
                
                news_id = f"buzzfeed-{counter}"
                outfile.write(f'{news_id},8,"{title}","{content}",{classification}\n')
            
            counter += 1

def main():
    base_dir = '.'  
    articles_dir = os.path.join(base_dir, 'articles')
    output_dir = os.path.join(base_dir, 'formated')
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'formatted_buzzfeed.csv')
    print("Processing BuzzFeed-Webis articles...")
    process_files(articles_dir, output_file)
    print("Completed processing articles")
    
    final_output = os.path.join(base_dir, 'buzzfeed-formated.csv')
    print("\nCreating final formatted file...")
    with open(output_file, 'r', encoding='utf-8') as infile, \
         open(final_output, 'w', encoding='utf-8') as outfile:
        outfile.write(infile.read())
    print("Created final file: buzzfeed-formated.csv")

if __name__ == '__main__':
    main() 