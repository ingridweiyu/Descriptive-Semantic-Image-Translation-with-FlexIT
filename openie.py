from openie import StanfordOpenIE
from sys import argv
import csv

# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.
properties = {
    'openie.affinity_probability_cap': 2 / 3,
}

with StanfordOpenIE(properties=properties) as client:
    text = argv[2]
    relations = ['replace', 'change', 'transform']
    subject_list = []
    object_list = [] 
    
    print('Text: %s.' % text)
    
    for triple in client.annotate(text):
        print('|-', triple)
        if any(r in triple['relation'] for r in relations):
            subject_list.append(triple['subject'])
            object_list.append(triple['object'])
    #print('subject list: ', subject_list)
    #print('object list: ', object_list)
    
    # data to be written row-wise in csv fil
    data = [subject_list, object_list]
  
    # opening the csv file in 'w+' mode
    filepath = argv[1]
    file = open(filepath, 'w+', newline ='')
      
    # writing the data into the file
    with file:    
        write = csv.writer(file)
        write.writerows(data)
    
