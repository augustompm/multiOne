# multi_memetic/utils/xml_parser.py

import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np

class ConservationLevel:
    """Define os níveis de conservação conforme BAliBASE4"""
    HIGH = 'HIGH'      # >25%
    MEDIUM = 'MEDIUM'  # 20-25% 
    LOW = 'LOW'       # <15%

class GlobalConservationMap:
    """Mantém estatísticas globais de conservação do Reference Set"""
    def __init__(self):
        self.block_stats = {
            ConservationLevel.HIGH: defaultdict(int),
            ConservationLevel.MEDIUM: defaultdict(int),
            ConservationLevel.LOW: defaultdict(int)
        }
        self.pattern_frequency = defaultdict(int)
        self.score_impact = defaultdict(list)
        
    def add_block_stats(self, level: str, block_data: Dict) -> None:
        """Adiciona estatísticas de um bloco"""
        self.block_stats[level]['count'] += 1
        self.block_stats[level]['total_length'] += block_data['length']
        self.block_stats[level]['total_score'] += block_data['score']
        
        # Registra padrões de aminoácidos se disponível
        if 'pattern' in block_data:
            self.pattern_frequency[block_data['pattern']] += 1
            
    def add_score_impact(self, level: str, score: float) -> None:
        """Registra impacto no score final"""
        self.score_impact[level].append(score)
        
    def get_summary(self) -> Dict:
        """Retorna sumário das estatísticas"""
        summary = {}
        for level in self.block_stats:
            stats = self.block_stats[level]
            if stats['count'] > 0:
                summary[level] = {
                    'count': stats['count'],
                    'avg_length': stats['total_length'] / stats['count'],
                    'avg_score': stats['total_score'] / stats['count'],
                    'score_impact': np.mean(self.score_impact[level])
                }
        return summary

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
        self.global_map = GlobalConservationMap()
        self.current_blocks = {
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
                            level = ConservationLevel.HIGH
                            self.conservation_blocks[ConservationLevel.HIGH].append(block_info)
                        elif 20.0 <= score <= 25.0:
                            level = ConservationLevel.MEDIUM
                            self.conservation_blocks[ConservationLevel.MEDIUM].append(block_info)
                        else:
                            level = ConservationLevel.LOW
                            self.conservation_blocks[ConservationLevel.LOW].append(block_info)
                        
                        # Mantém estruturas originais
                        self.block_scores[score].append(block_info)
                        self.color_mapping[color].add(score)

                    except Exception as e:
                        self.logger.debug(f"Skipping block with score {score}: {e}")
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

    def analyze_reference_set(self, xml_files: List[Path]) -> Dict:
        """Analisa todo o Reference Set 10"""
        self.logger.info(f"Analyzing {len(xml_files)} reference alignments...")
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Processa blocos do alinhamento atual
                self._process_alignment_blocks(root)
                
                # Avalia impacto no score final se disponível
                score_element = root.find(".//aln-score")
                if score_element is not None:
                    score = float(score_element.text)
                    self._evaluate_score_impact(score)
                    
            except Exception as e:
                self.logger.error(f"Error processing {xml_file}: {e}")
                continue
                
        return self.global_map.get_summary()
        
    def _process_alignment_blocks(self, root: ET.Element) -> None:
        """Processa blocos de um alinhamento"""
        self.current_blocks.clear()
        
        for seq in root.findall(".//sequence"):
            for block in seq.findall(".//fitem"):
                if block.find("ftype").text == "BLOCK":
                    try:
                        score = float(block.find("fscore").text)
                        block_data = {
                            'start': int(block.find("fstart").text),
                            'stop': int(block.find("fstop").text),
                            'score': score,
                            'length': int(block.find("fstop").text) - 
                                     int(block.find("fstart").text) + 1
                        }
                        
                        # Classifica e registra bloco
                        if score > 25.0:
                            level = ConservationLevel.HIGH
                        elif 20.0 <= score <= 25.0:
                            level = ConservationLevel.MEDIUM
                        elif score < 15.0:
                            level = ConservationLevel.LOW
                        else:
                            continue
                            
                        self.current_blocks[level].append(block_data)
                        self.global_map.add_block_stats(level, block_data)
                        
                    except Exception as e:
                        self.logger.debug(f"Skipping invalid block: {e}")
                        continue
                        
                elif block.find("ftype").text == "DISORDER":
                    try:
                        start = int(block.find("fstart").text)
                        stop = int(block.find("fstop").text)
                        self.disorder_regions.add((start, stop))
                    except Exception:
                        continue

    def _evaluate_score_impact(self, alignment_score: float) -> None:
        """Avalia impacto de cada nível no score final"""
        total_blocks = sum(len(blocks) for blocks in self.current_blocks.values())
        if total_blocks == 0:
            return
            
        # Estima contribuição proporcional
        for level, blocks in self.current_blocks.items():
            if blocks:
                impact = (len(blocks) / total_blocks) * alignment_score
                self.global_map.add_score_impact(level, impact)
                
    def get_blocks_by_conservation(self, level: str) -> List[Dict]:
        """Retorna blocos do nível especificado"""
        return self.current_blocks.get(level, [])