from typing import List, Optional, Tuple
from datasets import load_dataset
import re
import json

class QAData:
    def __init__(self, data_file_list, num_samples=500) -> None:
        """初始化HotpotQA数据类"""
        self.golden_answers = {}
        for data_file in data_file_list:
            test_dataset = load_dataset(
                'json',
                data_files=data_file,
                split="train",
            )
            num_flag = 0
            for s in test_dataset:
                self.golden_answers[s['question']] = s['golden_answers'][0]
                num_flag += 1
                if num_flag >= num_samples:
                    break

    def get_pairs(self) -> List[Tuple[str, str]]:
        """返回问题和答案对"""
        return list(self.golden_answers.items())

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """提取最后一个<answer>内容"""
        matches = re.findall(r'<answer>(.*?)</answer>', text)
        return matches[-1].strip() if matches else None

    def accuracy(self, outputs: List[Tuple[str, str]], single_answer=True) -> float:
        """计算准确度，仅考虑有答案的输出"""
        correct = 0
        total = 0
        for question, text in outputs:
            ans_start_idx = text.find(question) + len(question)
            if ans := QAData.extract_answer(text[ans_start_idx:]):
                total += 1
                if not single_answer:
                    continue
                if self.golden_answers[question].lower() in ans.lower():
                    correct += 1
                # logging.info(self.golden_answers[question].lower(), '||', ans.lower())
        return total, correct / len(outputs) if len(outputs) else 0.0
