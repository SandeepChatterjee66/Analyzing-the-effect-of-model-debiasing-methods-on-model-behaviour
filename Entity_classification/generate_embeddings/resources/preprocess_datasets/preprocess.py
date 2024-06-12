import spacy 
import os
import argparse
from tqdm import tqdm

input_path = './'

#___________Approach 3 - Comparing text rather than position________

special_sym = ['-', '+', '(', ')', '[', ']']
def convert_dataset_to_BIO_format(nlp, tokenizer, dataset, processed_file_path, is_medmention=False, pmid_file_path=""):
    """
    Convert a dataset to BIO format and saves the preprocessed data into a text file.

    Args:
        nlp: spaCy language model.
        tokenizer: spaCy tokenizer.
        dataset: Path to the dataset file.
        processed_file_path: Path where the processed file will be saved.
        is_medmention: Boolean flag indicating if the dataset is MedMentions.
        pmid_file_path: Path to the PMIDs file (used for MedMentions).

    Returns:
        None
    """
    if os.path.exists(processed_file_path):
        os.remove(processed_file_path)
        print(f'{processed_file_path} was removed.')
    dataset_name = (dataset).split('/')[1]
    given_mentions = []
    entity_mentions = []
    sentence = ""
    tokenized_sent = []
    tokenized_token = []
    entity = []
    entity_labels = []
    entity_cui_id = []
    bio_tags = []
    curr_pmid = ""
    pmid_lst = []
    mention_id = {}
    processed_file_name = processed_file_path.split('/')[-1]
    processed_file_folder = processed_file_path.split('/')[0:-1]
    print(processed_file_folder)
    print(processed_file_name)
    if is_medmention:
        with open(pmid_file_path, 'r') as fh:
            for line in fh:
                pmid_lst.append(line.strip())

    with open(dataset, 'r') as fh:
        for idx,line in enumerate(tqdm(fh, ncols = 100)):            
            #line containing title(t) and abstract(a)
            if '\t' not in line and '|t|' in line:
                parts = line.split('|t|')
                curr_pmid = parts[0].strip()
                if len(parts[-1]) > 0:
                    sentence  = sentence.strip(' ') + " " + parts[-1].rstrip('\n')
            if '\t' not in line and '|a|' in line:
                parts = line.split('|a|')
                curr_pmid = parts[0].strip()
                if len(parts[-1]) > 0:
                    sentence  = sentence.strip(' ') + " " + parts[-1].rstrip('\n')
            # lines containing mentions
            # given_mentions has all mentions for a t+a combine
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) <= 4:
                    continue
                given_mentions.append(parts[3].strip())
                if (dataset.split('/'))[1] == "NCBI_disease":
                    entity_mentions.append([int(parts[1].strip()), int(parts[2].strip()), 'Disease', (parts[5].split(',')[0]).strip()])
                else:
                    entity_mentions.append([int(parts[1].strip()), int(parts[2].strip()), (parts[4].split(',')[0]).strip(), (parts[5].split(',')[0]).strip()])
                mention_id[parts[3].strip()] = [(parts[4].split(',')[0]).strip(),(parts[5].split(',')[0]).strip()]
            
            # when one t+a finished reading
            start = 0
            if len(line.strip()) == 0:
                if is_medmention and curr_pmid not in pmid_lst:
                    given_mentions = []
                    sentence = ""
                    entity_mentions = []
                    tokenized_sent = []
                    tokenized_token = []
                    bio_tags = []
                if len(sentence) == 0:
                    continue
                for entity_mention in entity_mentions:
                    l = entity_mention[0] 
                    r = entity_mention[1] 
                    entity.append(sentence[l:r])
                    entity_labels.append(entity_mention[2])
                    entity_cui_id.append(entity_mention[3])

                    
                    sentence_wo_entity = sentence[start : l]
                    sentence_wo_entity = sentence_wo_entity.split(' ')
                    for token in sentence_wo_entity:
                        if len(token) > 0 and token != ' ':
                            tokenized_sent.append((str(token)).strip(' '))
                            bio_tags.append('O')
                    sentence_w_entity = sentence[l:r]
                    tokenized_sent.append((str(sentence_w_entity)).strip(' '))
                    bio_tags.append('B' + '-' + entity_mention[2])
                    start = r
                sentence_wo_entity = ""
                for i in range(start, len(sentence)):
                    sentence_wo_entity += sentence[i]
                sentence_wo_entity = sentence_wo_entity.split(' ')
                for token in sentence_wo_entity:
                    if len(token) > 0 and token != ' ':
                        tokenized_sent.append((str(token)).strip(' '))
                        bio_tags.append('O')


                with open(processed_file_path, 'a') as fh:
                    for token, tag in zip(tokenized_sent, bio_tags):
                            fh.write(f'{token}\t{tag}\n')
                    fh.write('\n')

                given_mentions = []
                sentence = ""
                tokenized_sent = []
                tokenized_token = []
                entity_mentions = []
                bio_tags = []
                mention_id = {}
    cui_file_name = 'entity_cui_' + processed_file_name
    cui_path = os.path.join(input_path, processed_file_folder[0], processed_file_folder[1], cui_file_name)
    print(cui_path)
    with open(cui_path, 'w') as fh:
        for i in range(len(entity)):
            fh.write(f'{entity[i]}\t{entity_labels[i]}\t{entity_cui_id[i]}\n')
    print(len(entity))

def main():
    """
    Main function to parse arguments and start the preprocessing of the data.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pmid_file', type=str, required=False)#____used for MedMentions_____
    parser.add_argument('--processed_file_path', type=str, required=True)

    args = parser.parse_args()

    #tokenizer for tokenizing the dataset
    nlp = spacy.load('en_core_web_trf')
    tokenizer = nlp.tokenizer

    #for preprocessing the text
    dataset_name = (args.dataset).split('/')[1]
    
    if dataset_name == 'MedMentions':
        print('MedMentions')
        convert_dataset_to_BIO_format(nlp, tokenizer, args.dataset, args.processed_file_path, is_medmention=True, pmid_file_path=args.pmid_file)
    else:
        convert_dataset_to_BIO_format(nlp, tokenizer, args.dataset, args.processed_file_path)

if __name__ == '__main__':
    main()

