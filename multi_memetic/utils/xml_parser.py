# multi_memetic/utils/xml_parser.py

import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

class ConservationLevel:
    """Define os níveis de conservação conforme BAliBASE4"""
    HIGH = 'HIGH'      # >25%
    MEDIUM = 'MEDIUM'  # 20-25% 
    LOW = 'LOW'       # <15%

class ScoreAccessLayer:
    """Adaptação do ScoreAccessLayer original para classificar por níveis de conservação"""
    
    def __init__(self, logger=None):
        self.column_scores = {}
        self.column_score_owners = {}
        self.group_score_mapping = {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.block_scores = defaultdict(list)
        self.color_mapping = defaultdict(set)
        self.disorder_regions = set()
        
        # Novo: classificação de blocos por nível de conservação
        self.conservation_blocks = {
            ConservationLevel.HIGH: [],
            ConservationLevel.MEDIUM: [],
            ConservationLevel.LOW: []
        }

    def load_from_xml(self, root) -> None:
        """Carrega e classifica blocos do XML mantendo lógica original"""
        self.column_scores.clear()
        self.column_score_owners.clear()
        self.group_score_mapping.clear()
        self.block_scores.clear()
        self.color_mapping.clear()
        self.disorder_regions.clear()
        
        for level in self.conservation_blocks:
            self.conservation_blocks[level].clear()

        # Mantém lógica original de carregar column-scores
        for colsco in root.findall(".//column-score"):
            try:
                name = colsco.find("colsco-name").text
                owner = colsco.find("colsco-owner").text if colsco.find("colsco-owner") is not None else ""
                data_text = colsco.find("colsco-data").text
                if not data_text:
                    continue

                data = [int(x) for x in data_text.split()]
                self.column_scores[name] = data

                if owner:
                    self.column_score_owners[name] = owner

                if "normd_" in name or "group" in owner:
                    try:
                        group_id = int(''.join(filter(str.isdigit, name if "normd_" in name else owner)))
                        self.group_score_mapping[group_id] = name
                    except ValueError:
                        continue

            except Exception as e:
                self.logger.debug(f"Skipping score {name}: {e}")
                continue

        # Carrega blocos e classifica por conservação
        for seq in root.findall(".//sequence"):
            for block in seq.findall(".//fitem"):
                if block.find("ftype").text == "BLOCK":
                    try:
                        score = float(block.find("fscore").text)
                        color = int(block.find("fcolor").text)
                        start = int(block.find("fstart").text)
                        stop = int(block.find("fstop").text)

                        block_info = {
                            'start': start,
                            'stop': stop,
                            'color': color,
                            'length': stop - start + 1,
                            'score': score
                        }
                        
                        # Classifica conforme documentação do BAliBASE4
                        if score > 25.0:
                            self.conservation_blocks[ConservationLevel.HIGH].append(block_info)
                        elif 20.0 <= score <= 25.0:
                            self.conservation_blocks[ConservationLevel.MEDIUM].append(block_info)
                        else:
                            self.conservation_blocks[ConservationLevel.LOW].append(block_info)

                        # Mantém estruturas originais
                        self.block_scores[score].append(block_info)
                        self.color_mapping[color].add(score)

                    except Exception:
                        continue

                elif block.find("ftype").text == "DISORDER":
                    try:
                        start = int(block.find("fstart").text)
                        stop = int(block.find("fstop").text)
                        self.disorder_regions.add((start, stop))
                    except Exception:
                        continue

    def get_block_boundaries(self, min_score: float, conservation_level: str = None) -> List[Tuple[int, int]]:
        """Retorna limites dos blocos filtrados por score e nível de conservação"""
        boundaries = []
        
        if conservation_level:
            blocks = self.conservation_blocks[conservation_level]
            for block in blocks:
                if block['score'] >= min_score:
                    boundaries.append((block['start'], block['stop']))
        else:
            # Comportamento original
            for score, blocks in self.block_scores.items():
                if score >= min_score:
                    for block in blocks:
                        boundaries.append((block['start'], block['stop']))
                        
        return sorted(boundaries)

    def get_blocks_by_conservation(self, level: str) -> List[Dict]:
        """Retorna blocos para um determinado nível de conservação"""
        if not self.conservation_blocks:
            self.logger.error("Conservation blocks not initialized")
            return []
            
        if level not in self.conservation_blocks:
            self.logger.error(f"Invalid conservation level: {level}")
            return []
            
        return self.conservation_blocks[level]