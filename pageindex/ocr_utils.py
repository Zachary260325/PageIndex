"""
OCR utilities for processing external OCR results instead of PyMuPDF.
This module provides functions to work with OCR JSON data and extract
table of contents with bbox information for indexing.
"""

import json
import tiktoken
import re
from typing import List, Dict, Any, Tuple, Optional


def load_ocr_results(ocr_file_path: str) -> List[Dict[str, Any]]:
    """
    Load OCR results from JSON file.
    
    Args:
        ocr_file_path: Path to the OCR JSON file
        
    Returns:
        List of page dictionaries with OCR data
    """
    with open(ocr_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_text_from_ocr_page(page_data: Dict[str, Any]) -> str:
    """
    Extract text content from a single OCR page.
    
    Args:
        page_data: OCR data for a single page
        
    Returns:
        Concatenated text content from the page
    """
    texts = []
    if 'full_layout_info' in page_data:
        for item in page_data['full_layout_info']:
            if 'text' in item and item['text'].strip():
                texts.append(item['text'].strip())
    
    return '\n'.join(texts)


def get_page_tokens_from_ocr(ocr_results: List[Dict[str, Any]], model: str = "gpt-4o-2024-11-20") -> List[Tuple[str, int]]:
    """
    Convert OCR results to page_list format expected by existing code.
    
    Args:
        ocr_results: List of OCR page data
        model: Model name for token counting
        
    Returns:
        List of tuples (page_text, token_count)
    """
    enc = tiktoken.encoding_for_model(model)
    page_list = []
    
    for page_data in ocr_results:
        page_text = extract_text_from_ocr_page(page_data)
        token_length = len(enc.encode(page_text))
        page_list.append((page_text, token_length))
    
    return page_list


def extract_section_headers_with_bbox(ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract section headers with their bbox information from OCR results.
    
    Args:
        ocr_results: List of OCR page data
        
    Returns:
        List of section headers with page number, text, bbox, and category
    """
    headers = []
    
    for page_data in ocr_results:
        page_no = page_data.get('page_no', 0)
        
        if 'full_layout_info' in page_data:
            for item in page_data['full_layout_info']:
                # Look for section headers
                if ('category' in item and 
                    'Section-header' in item.get('category', '') and
                    'text' in item and 
                    'bbox' in item):
                    
                    headers.append({
                        'page_no': page_no,
                        'text': item['text'].strip(),
                        'bbox': item['bbox'],
                        'category': item['category'],
                        'physical_index': page_no + 1  # Convert to 1-based indexing
                    })
    
    return headers


def detect_toc_from_headers(headers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Try to detect table of contents structure from section headers.
    
    Args:
        headers: List of section headers with bbox info
        
    Returns:
        Dictionary with toc_content and structure information
    """
    toc_items = []
    
    for header in headers:
        text = header['text']
        
        # Remove markdown-style headers and clean text
        cleaned_text = re.sub(r'^#+\s*', '', text).strip()
        
        # Try to determine hierarchy level from markdown headers or text patterns
        level = 1
        if text.startswith('####'):
            level = 4
        elif text.startswith('###'):
            level = 3
        elif text.startswith('##'):
            level = 2
        elif text.startswith('#'):
            level = 1
        
        # Generate structure index based on level and position
        structure_parts = []
        if level == 1:
            structure_parts = [str(len([h for h in toc_items if h.get('level', 1) == 1]) + 1)]
        elif level == 2:
            parent_sections = [h for h in toc_items if h.get('level', 1) == 1]
            current_l2_count = len([h for h in toc_items if h.get('level', 1) == 2 and 
                                  h.get('parent_section') == len(parent_sections)])
            structure_parts = [str(len(parent_sections)), str(current_l2_count + 1)]
        # Add more levels as needed
        
        structure = '.'.join(structure_parts) if structure_parts else None
        
        toc_item = {
            'structure': structure,
            'title': cleaned_text,
            'physical_index': header['physical_index'],
            'bbox': header['bbox'],
            'page_no': header['page_no'],
            'level': level
        }
        
        if level == 2 and len([h for h in toc_items if h.get('level', 1) == 1]) > 0:
            toc_item['parent_section'] = len([h for h in toc_items if h.get('level', 1) == 1])
        
        toc_items.append(toc_item)
    
    return {
        'toc_items': toc_items,
        'has_structure': len(toc_items) > 0
    }


def find_text_bbox_in_page(page_data: Dict[str, Any], search_text: str, fuzzy_match: bool = True) -> Optional[List[int]]:
    """
    Find the bbox coordinates of specific text in a page.
    
    Args:
        page_data: OCR data for a single page
        search_text: Text to search for
        fuzzy_match: Whether to perform fuzzy matching
        
    Returns:
        Bbox coordinates [x1, y1, x2, y2] if found, None otherwise
    """
    if 'full_layout_info' not in page_data:
        return None
    
    search_text_clean = re.sub(r'\s+', ' ', search_text.lower().strip())
    
    for item in page_data['full_layout_info']:
        if 'text' in item and 'bbox' in item:
            item_text_clean = re.sub(r'\s+', ' ', item['text'].lower().strip())
            
            if fuzzy_match:
                # Check if the search text is contained in the item text or vice versa
                if (search_text_clean in item_text_clean or 
                    item_text_clean in search_text_clean or
                    # Check for substantial overlap
                    len(set(search_text_clean.split()) & set(item_text_clean.split())) > 
                    min(len(search_text_clean.split()), len(item_text_clean.split())) * 0.5):
                    return item['bbox']
            else:
                if search_text_clean == item_text_clean:
                    return item['bbox']
    
    return None


def enhance_toc_with_bbox(toc_items: List[Dict[str, Any]], ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance TOC items with bbox information by searching in OCR results.
    
    Args:
        toc_items: List of TOC items 
        ocr_results: OCR results data
        
    Returns:
        Enhanced TOC items with bbox information
    """
    enhanced_items = []
    
    for item in toc_items:
        enhanced_item = item.copy()
        
        # Try to find bbox for this TOC item
        if 'physical_index' in item and 'title' in item:
            physical_index = item['physical_index']
            title = item['title']
            
            # Find the corresponding page in OCR results (convert to 0-based indexing)
            page_index = physical_index - 1
            if 0 <= page_index < len(ocr_results):
                page_data = ocr_results[page_index]
                bbox = find_text_bbox_in_page(page_data, title)
                if bbox:
                    enhanced_item['bbox'] = bbox
                    enhanced_item['page_no'] = page_data.get('page_no', page_index)
        
        enhanced_items.append(enhanced_item)
    
    return enhanced_items


def get_text_of_ocr_pages(ocr_results: List[Dict[str, Any]], start_page: int, end_page: int) -> str:
    """
    Get text content from a range of OCR pages.
    
    Args:
        ocr_results: OCR results data
        start_page: Start page number (1-based)
        end_page: End page number (1-based, inclusive)
        
    Returns:
        Concatenated text from the specified pages
    """
    text = ""
    for page_num in range(start_page - 1, end_page):
        if 0 <= page_num < len(ocr_results):
            page_text = extract_text_from_ocr_page(ocr_results[page_num])
            text += page_text + "\n"
    
    return text


def get_text_of_ocr_pages_with_labels(ocr_results: List[Dict[str, Any]], start_page: int, end_page: int) -> str:
    """
    Get text content from a range of OCR pages with physical index labels.
    
    Args:
        ocr_results: OCR results data
        start_page: Start page number (1-based)
        end_page: End page number (1-based, inclusive)
        
    Returns:
        Concatenated text with page labels from the specified pages
    """
    text = ""
    for page_num in range(start_page - 1, end_page):
        if 0 <= page_num < len(ocr_results):
            page_text = extract_text_from_ocr_page(ocr_results[page_num])
            text += f"<physical_index_{page_num + 1}>\n{page_text}\n<physical_index_{page_num + 1}>\n"
    
    return text


def create_ocr_page_list_wrapper(ocr_results: List[Dict[str, Any]], model: str = "gpt-4o-2024-11-20"):
    """
    Create a wrapper that makes OCR results compatible with existing page_list format
    while preserving access to bbox information.
    
    Args:
        ocr_results: OCR results data
        model: Model name for token counting
        
    Returns:
        OCRPageListWrapper object that behaves like a page_list but preserves OCR data
    """
    return OCRPageListWrapper(ocr_results, model)


class OCRPageListWrapper:
    """
    Wrapper class that makes OCR results behave like the traditional page_list
    while preserving access to the original OCR data with bbox information.
    """
    
    def __init__(self, ocr_results: List[Dict[str, Any]], model: str = "gpt-4o-2024-11-20"):
        self.ocr_results = ocr_results
        self.model = model
        self._page_list = get_page_tokens_from_ocr(ocr_results, model)
    
    def __len__(self) -> int:
        return len(self._page_list)
    
    def __getitem__(self, index) -> Tuple[str, int]:
        return self._page_list[index]
    
    def __iter__(self):
        return iter(self._page_list)
    
    def get_ocr_data(self, page_index: int) -> Optional[Dict[str, Any]]:
        """Get the original OCR data for a specific page."""
        if 0 <= page_index < len(self.ocr_results):
            return self.ocr_results[page_index]
        return None
    
    def find_text_bbox(self, page_index: int, text: str) -> Optional[List[int]]:
        """Find bbox coordinates for text on a specific page."""
        ocr_data = self.get_ocr_data(page_index)
        if ocr_data:
            return find_text_bbox_in_page(ocr_data, text)
        return None
    
    def get_all_bboxes_for_page(self, page_index: int) -> List[Dict[str, Any]]:
        """Get all text items with bbox information for a page."""
        ocr_data = self.get_ocr_data(page_index)
        if not ocr_data or 'full_layout_info' not in ocr_data:
            return []
        
        return [item for item in ocr_data['full_layout_info'] 
                if 'bbox' in item and 'text' in item]
