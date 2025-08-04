import roman
from typing import Dict, List, Tuple, Optional


class TestPageNumberMapping:
    """Test logical page number mapping accuracy."""
    
    @staticmethod
    def get_expected_mapping() -> Dict[int, Optional[str]]:
        """Generate the expected mapping from the CLAUDE.md file."""
        return {
            pdf_page: (
                # Pages 3-7: lowercase roman numerals i-v
                roman.toRoman(pdf_page - 2).lower() if 3 <= pdf_page <= 7
                # Pages 9-15: uppercase roman numerals I-VII
                else roman.toRoman(pdf_page - 8) if 9 <= pdf_page <= 15
                # Pages 17-67: Arabic integers 1-51
                else str(pdf_page - 16) if 17 <= pdf_page <= 67
                # Pages 68-70: capital letters A-C
                else chr(ord('A') + pdf_page - 68) if 68 <= pdf_page <= 70
                # All other pages: None
                else None
            )
            for pdf_page in range(1, 73)
        }
    
    @staticmethod
    def extract_actual_mapping(actual_data: List[dict]) -> Dict[int, Optional[str]]:
        """Extract page number mapping from actual results."""
        actual_mapping = {}
        for block in actual_data:
            page_num = block["page_number"]
            logical_page = block["logical_page_number"]
            if page_num not in actual_mapping:
                actual_mapping[page_num] = logical_page
        return actual_mapping
    
    @staticmethod
    def analyze_errors(
        expected_mapping: Dict[int, Optional[str]], 
        actual_mapping: Dict[int, Optional[str]]
    ) -> Tuple[int, int, List[Tuple[int, Optional[str], Optional[str]]]]:
        """Analyze errors between expected and actual mappings."""
        total = 72
        correct = 0
        errors = []
        
        for pdf_page in range(1, 73):
            expected = expected_mapping[pdf_page]
            actual = actual_mapping.get(pdf_page)
            
            if expected == actual:
                correct += 1
            else:
                errors.append((pdf_page, expected, actual))
        
        return correct, total, errors
    
    @staticmethod
    def log_detailed_results(correct: int, total: int, errors: List[Tuple[int, Optional[str], Optional[str]]]):
        """Log detailed results for debugging purposes."""
        print("\nDETAILED RESULTS")
        print("=" * 50)
        print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"Errors: {len(errors)}")
        
        if errors:
            print("\nDETAILED ERRORS:")
            print("PDF Page | Expected | Actual")
            print("-" * 30)
            for pdf_page, expected, actual in errors:
                print(f"{pdf_page:8} | {str(expected):8} | {str(actual)}")
    
    @staticmethod
    def log_section_analysis(
        expected_mapping: Dict[int, Optional[str]], 
        actual_mapping: Dict[int, Optional[str]]
    ):
        """Log accuracy by document section."""
        print("\nSUMMARY BY SECTION:")
        sections = [
            ("Pages 1-2 (cover)", range(1, 3)),
            ("Pages 3-7 (roman i-v)", range(3, 8)),
            ("Page 8 (separator)", [8]),
            ("Pages 9-15 (roman I-VII)", range(9, 16)),
            ("Page 16 (separator)", [16]),
            ("Pages 17-67 (numbers 1-51)", range(17, 68)),
            ("Pages 68-70 (letters A-C)", range(68, 71)),
            ("Pages 71-72 (end matter)", range(71, 73))
        ]
        
        for section_name, page_range in sections:
            section_correct = 0
            section_total = len(list(page_range))
            for pdf_page in page_range:
                if expected_mapping[pdf_page] == actual_mapping.get(pdf_page):
                    section_correct += 1
            accuracy = section_correct/section_total*100 if section_total > 0 else 0
            print(f"{section_name}: {section_correct}/{section_total} ({accuracy:.1f}%)")
    
    def test_page_number_mapping_accuracy(self, doc_601_with_logical_pages):
        """Test that logical page number mapping achieves at least 95% accuracy."""
        # Get expected mapping
        expected_mapping = self.get_expected_mapping()
        
        # Extract actual mapping from test data
        actual_mapping = self.extract_actual_mapping(doc_601_with_logical_pages)
        
        # Analyze results
        correct, total, errors = self.analyze_errors(expected_mapping, actual_mapping)
        accuracy = correct / total
        
        # Log detailed results for debugging
        self.log_detailed_results(correct, total, errors)
        self.log_section_analysis(expected_mapping, actual_mapping)
        
        # Assert minimum accuracy requirement
        assert accuracy >= 0.95, (
            f"Page number mapping accuracy {accuracy:.1%} is below required 95%. "
            f"Got {correct}/{total} correct with {len(errors)} errors."
        )
    
    def test_expected_mapping_structure(self):
        """Verify the expected mapping has the correct structure."""
        expected_mapping = self.get_expected_mapping()
        
        # Should have entries for pages 1-72
        assert len(expected_mapping) == 72
        assert all(page in expected_mapping for page in range(1, 73))
        
        # Test specific expected values
        assert expected_mapping[1] is None  # Cover page
        assert expected_mapping[3] == "i"   # First roman lowercase
        assert expected_mapping[9] == "I"   # First roman uppercase
        assert expected_mapping[17] == "1"  # First arabic numeral
        assert expected_mapping[68] == "A"  # First letter
        assert expected_mapping[72] is None # End matter