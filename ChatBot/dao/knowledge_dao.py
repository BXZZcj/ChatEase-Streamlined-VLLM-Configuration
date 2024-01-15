import sys
import os
# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)
# 获取当前脚本的父目录的父目录（即 config 和 llm 所在的目录）
project_dir = os.path.dirname(os.path.dirname(script_path))
# 将 project_dir 添加到 sys.path
if project_dir not in sys.path:
    sys.path.append(project_dir)


import os
from docx import Document
import unicodedata
import pickle
from util.embedding import calculate_embedding


class KnowledgeDAO:
    def __init__(self):
        self.knowledge_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'knowledge')
        self.text_flag = {
            'question_type': False,
            'question': False,
            'answer': False
        }

    def identify_text_flag(self, text):
        if text == '':
            ...
        if len(text) >= 2 and text[0].isdigit() and text[1] == '.':
            self.text_flag['question_type'] = True
            self.text_flag['question'] = False
            self.text_flag['answer'] = False
        elif (len(text) >= 2 and text[0] == 'Q' and text[1] == ':') or (len(text) >= 4 and text[0].isdigit() and text[1] == '、' and text[2] == 'Q'):
            self.text_flag['question_type'] = False
            self.text_flag['question'] = True
            self.text_flag['answer'] = False
        elif self.text_flag['question'] and not self.text_flag['answer']:
            self.text_flag['question'] = False
            self.text_flag['answer'] = True

    def knowledge_to_kg(self):
        knowledge_doc = Document(f'{self.knowledge_data_path}/DM常见问题.docx')
        question_set, ontology_triple_list, knowledge_triple_list = set(), [], []

        temp_dict = dict()
        for paragraph in knowledge_doc.paragraphs:
            text = unicodedata.normalize('NFKC', paragraph.text.strip())
            self.identify_text_flag(text)
            if self.text_flag['question_type']:
                if 'question' in temp_dict.keys():
                    question_set.add(temp_dict['question'])
                    knowledge_triple_list.append((temp_dict['question'], 'answer_is', temp_dict['answer_temp']))
                    ontology_triple_list.append((temp_dict['question_type'], 'is_a_subclass_of', 'question'))
                    ontology_triple_list.append((temp_dict['question'], 'is_an_instance_of', temp_dict['question_type']))
                    ontology_triple_list.append((temp_dict['answer_temp'], 'is_an_instance_of', 'answer'))
                temp_dict = {
                    'question_type': text.split('.')[1],
                    'answer_temp': ''
                }
            elif self.text_flag['question']:
                if 'question' in temp_dict.keys():
                    question_set.add(temp_dict['question'])
                    knowledge_triple_list.append((temp_dict['question'], 'answer_is', temp_dict['answer_temp']))
                    ontology_triple_list.append((temp_dict['question_type'], 'is_a_subclass_of', 'question'))
                    ontology_triple_list.append((temp_dict['question'], 'is_an_instance_of', temp_dict['question_type']))
                    ontology_triple_list.append((temp_dict['answer_temp'], 'is_an_instance_of', 'answer'))
                # if '升级后登录' in text:
                #     print(text)
                if 'Q:' in text:
                    temp_dict['question'] = text.split('Q:')[1]
                else:
                    temp_dict['question'] = text.split('、Q')[1]
                temp_dict['answer_temp'] = ''
            elif self.text_flag['answer']:
                if temp_dict['answer_temp'] != '':
                    temp_dict['answer_temp'] += '  '
                if 'A:' in text:
                    temp_dict['answer_temp'] += text.split('A:')[1]
                else:
                    temp_dict['answer_temp'] += text
        question_list = list(question_set)
        question_embedding_list = calculate_embedding(question_list)

        with open(f'{self.knowledge_data_path}/extracted_qa_knowledge.pkl', 'wb') as f:
            pickle.dump({
                'knowledge': knowledge_triple_list,
                'ontology': ontology_triple_list,
                'question_list': question_list,
                'question_embeddings': question_embedding_list
            }, f)

    def load_qa_knowledge(self):
        with open(f'{self.knowledge_data_path}/extracted_qa_knowledge.pkl', 'rb') as f:
            qa_knowledge = pickle.load(f)
        return qa_knowledge


if __name__ == '__main__':
    knowledge_dao = KnowledgeDAO()
    knowledge_dao.knowledge_to_kg()
    a = knowledge_dao.load_qa_knowledge()
    ...
