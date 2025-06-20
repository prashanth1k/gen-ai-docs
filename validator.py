"""
Validation module for gen-ai-docs CLI tool.
Analyzes generated markdown against original PDF to provide efficiency and coverage statistics.
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import Counter


def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """
    Extract content from the original PDF for comparison.
    
    Args:
        pdf_path (str): Path to the original PDF file
        
    Returns:
        Dict containing extracted content statistics
    """
    try:
        # Use the same PDF parser as the main tool
        import pdf_parser
        
        print(f"📄 Extracting content from original PDF: {pdf_path}")
        
        # Parse PDF using ultra-fast mode
        elements = pdf_parser.parse_pdf(pdf_path, enhanced_mode=False, hi_res=False)
        
        # Extract text content
        all_text = []
        for elem in elements:
            if hasattr(elem, 'text') and elem.text:
                all_text.append(elem.text.strip())
        
        full_text = " ".join(all_text)
        
        # Basic text analysis
        words = full_text.lower().split()
        sentences = re.split(r'[.!?]+', full_text)
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        
        # Extract potential topics/headings (lines that are short and title-case)
        potential_headings = []
        for elem in elements:
            if hasattr(elem, 'text') and elem.text:
                text = elem.text.strip()
                if (len(text) < 100 and 
                    len(text.split()) <= 8 and 
                    any(word[0].isupper() for word in text.split() if word)):
                    potential_headings.append(text)
        
        # Extract key terms (words that appear frequently and are longer)
        word_freq = Counter(word for word in words if len(word) > 4)
        key_terms = [word for word, count in word_freq.most_common(50)]
        
        return {
            'total_elements': len(elements),
            'total_characters': len(full_text),
            'total_words': len(words),
            'total_sentences': len([s for s in sentences if s.strip()]),
            'total_paragraphs': len(paragraphs),
            'potential_headings': potential_headings,
            'key_terms': key_terms,
            'word_frequency': dict(word_freq.most_common(20)),
            'full_text': full_text
        }
        
    except Exception as e:
        raise Exception(f"Failed to extract PDF content: {str(e)}")


def analyze_markdown_content(markdown_path: str) -> Dict[str, Any]:
    """
    Analyze the generated markdown file structure and content.
    
    Args:
        markdown_path (str): Path to the generated markdown file
        
    Returns:
        Dict containing markdown analysis
    """
    try:
        print(f"📝 Analyzing generated markdown: {markdown_path}")
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract structure
        h1_headers = re.findall(r'^# (.+)$', content, re.MULTILINE)
        h2_headers = re.findall(r'^## (.+)$', content, re.MULTILINE)
        h3_headers = re.findall(r'^### (.+)$', content, re.MULTILINE)
        
        # Extract features
        features = re.findall(r'^### Feature: (.+)$', content, re.MULTILINE)
        
        # Extract descriptions, functionality, and use cases
        descriptions = re.findall(r'- \*\*Description:\*\* (.+)', content)
        functionality_sections = re.findall(r'- \*\*Key Functionality:\*\*\n((?:  - .+\n?)*)', content)
        use_cases = re.findall(r'- \*\*Use Case:\*\* (.+)', content)
        
        # Count functionality items
        total_functionality_items = 0
        for section in functionality_sections:
            items = re.findall(r'  - (.+)', section)
            total_functionality_items += len(items)
        
        # Basic text analysis
        words = content.lower().split()
        sentences = re.split(r'[.!?]+', content)
        
        # Extract all content text (without markdown formatting)
        clean_content = re.sub(r'[#*\-]', '', content)
        clean_content = re.sub(r'\n+', ' ', clean_content).strip()
        
        return {
            'total_characters': len(content),
            'clean_characters': len(clean_content),
            'total_words': len(words),
            'total_sentences': len([s for s in sentences if s.strip()]),
            'h1_count': len(h1_headers),
            'h2_count': len(h2_headers),
            'h3_count': len(h3_headers),
            'feature_count': len(features),
            'description_count': len(descriptions),
            'functionality_items': total_functionality_items,
            'use_case_count': len(use_cases),
            'h1_headers': h1_headers,
            'h2_headers': h2_headers,
            'h3_headers': h3_headers,
            'features': features,
            'clean_content': clean_content,
            'full_content': content
        }
        
    except Exception as e:
        raise Exception(f"Failed to analyze markdown: {str(e)}")


def calculate_coverage_metrics(pdf_data: Dict, markdown_data: Dict) -> Dict[str, Any]:
    """
    Calculate coverage and efficiency metrics between PDF and markdown.
    
    Args:
        pdf_data: Extracted PDF content data
        markdown_data: Analyzed markdown data
        
    Returns:
        Dict containing coverage metrics
    """
    print(f"🔍 Calculating coverage metrics...")
    
    # Content coverage (character-based)
    content_coverage = (markdown_data['clean_characters'] / pdf_data['total_characters']) * 100
    
    # Word coverage
    pdf_words = set(pdf_data['full_text'].lower().split())
    markdown_words = set(markdown_data['clean_content'].lower().split())
    word_coverage = (len(markdown_words & pdf_words) / len(pdf_words)) * 100 if pdf_words else 0
    
    # Key terms coverage
    pdf_key_terms = set(pdf_data['key_terms'])
    markdown_key_terms = set(word.lower() for word in markdown_data['clean_content'].split() if len(word) > 4)
    key_terms_coverage = (len(pdf_key_terms & markdown_key_terms) / len(pdf_key_terms)) * 100 if pdf_key_terms else 0
    
    # Heading coverage (approximate)
    pdf_headings = set(h.lower() for h in pdf_data['potential_headings'])
    markdown_headings = set(h.lower() for h in markdown_data['h1_headers'] + markdown_data['h2_headers'] + markdown_data['h3_headers'])
    
    # Find similar headings (fuzzy matching)
    heading_matches = 0
    for pdf_heading in pdf_headings:
        for md_heading in markdown_headings:
            # Simple similarity check - if 60% of words match
            pdf_words = set(pdf_heading.split())
            md_words = set(md_heading.split())
            if pdf_words and len(pdf_words & md_words) / len(pdf_words) >= 0.6:
                heading_matches += 1
                break
    
    heading_coverage = (heading_matches / len(pdf_headings)) * 100 if pdf_headings else 0
    
    # Structure completeness
    structure_score = 0
    if markdown_data['h1_count'] >= 1:  # Has main topic
        structure_score += 25
    if markdown_data['h2_count'] >= 2:  # Has capabilities
        structure_score += 25
    if markdown_data['feature_count'] >= 3:  # Has features
        structure_score += 25
    if markdown_data['description_count'] >= markdown_data['feature_count']:  # All features have descriptions
        structure_score += 25
    
    # Efficiency metrics
    compression_ratio = (pdf_data['total_characters'] / markdown_data['total_characters']) if markdown_data['total_characters'] > 0 else 0
    information_density = markdown_data['feature_count'] / (markdown_data['total_words'] / 1000) if markdown_data['total_words'] > 0 else 0
    
    return {
        'content_coverage_percent': round(content_coverage, 2),
        'word_coverage_percent': round(word_coverage, 2),
        'key_terms_coverage_percent': round(key_terms_coverage, 2),
        'heading_coverage_percent': round(heading_coverage, 2),
        'structure_completeness_percent': structure_score,
        'compression_ratio': round(compression_ratio, 2),
        'information_density': round(information_density, 2),
        'pdf_headings_found': len(pdf_headings),
        'markdown_headings_created': len(markdown_headings),
        'heading_matches': heading_matches,
        'missing_key_terms': list(pdf_key_terms - markdown_key_terms)[:10],  # Top 10 missing terms
        'unique_markdown_terms': list(markdown_key_terms - pdf_key_terms)[:10]  # Top 10 new terms
    }


def generate_validation_report(pdf_path: str, markdown_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Generate a comprehensive validation report comparing PDF to markdown.
    
    Args:
        pdf_path: Path to original PDF file
        markdown_path: Path to generated markdown file
        output_path: Optional path to save detailed report
        
    Returns:
        Dict containing complete validation results
    """
    print(f"🔍 Starting validation analysis...")
    print(f"📄 PDF: {pdf_path}")
    print(f"📝 Markdown: {markdown_path}")
    print()
    
    try:
        # Extract data from both sources
        pdf_data = extract_pdf_content(pdf_path)
        markdown_data = analyze_markdown_content(markdown_path)
        
        # Calculate metrics
        coverage_metrics = calculate_coverage_metrics(pdf_data, markdown_data)
        
        # Compile full report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'input_files': {
                'pdf_path': pdf_path,
                'markdown_path': markdown_path
            },
            'pdf_analysis': {
                'file_size_mb': round(os.path.getsize(pdf_path) / (1024 * 1024), 2),
                'total_elements': pdf_data['total_elements'],
                'total_characters': pdf_data['total_characters'],
                'total_words': pdf_data['total_words'],
                'total_sentences': pdf_data['total_sentences'],
                'potential_headings': len(pdf_data['potential_headings']),
                'key_terms_identified': len(pdf_data['key_terms'])
            },
            'markdown_analysis': {
                'file_size_kb': round(os.path.getsize(markdown_path) / 1024, 2),
                'total_characters': markdown_data['total_characters'],
                'clean_characters': markdown_data['clean_characters'],
                'total_words': markdown_data['total_words'],
                'structure': {
                    'main_topics': markdown_data['h1_count'],
                    'capabilities': markdown_data['h2_count'],
                    'features': markdown_data['feature_count'],
                    'descriptions': markdown_data['description_count'],
                    'functionality_items': markdown_data['functionality_items'],
                    'use_cases': markdown_data['use_case_count']
                }
            },
            'coverage_metrics': coverage_metrics,
            'quality_assessment': {
                'overall_score': round((
                    coverage_metrics['content_coverage_percent'] * 0.3 +
                    coverage_metrics['key_terms_coverage_percent'] * 0.3 +
                    coverage_metrics['structure_completeness_percent'] * 0.4
                ), 2),
                'content_preservation': 'Excellent' if coverage_metrics['content_coverage_percent'] >= 80 else 
                                     'Good' if coverage_metrics['content_coverage_percent'] >= 60 else
                                     'Fair' if coverage_metrics['content_coverage_percent'] >= 40 else 'Poor',
                'structure_quality': 'Excellent' if coverage_metrics['structure_completeness_percent'] == 100 else
                                   'Good' if coverage_metrics['structure_completeness_percent'] >= 75 else
                                   'Fair' if coverage_metrics['structure_completeness_percent'] >= 50 else 'Poor'
            }
        }
        
        # Save detailed report if requested
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📊 Detailed report saved to: {output_path}")
        
        return report
        
    except Exception as e:
        raise Exception(f"Validation failed: {str(e)}")


def print_validation_summary(report: Dict[str, Any]):
    """
    Print a formatted summary of the validation results.
    
    Args:
        report: Validation report dictionary
    """
    print("=" * 80)
    print("📊 VALIDATION REPORT SUMMARY")
    print("=" * 80)
    
    # Basic info
    print(f"📄 Original PDF: {report['input_files']['pdf_path']}")
    print(f"📝 Generated Markdown: {report['input_files']['markdown_path']}")
    print(f"⏰ Analysis Time: {report['validation_timestamp']}")
    print()
    
    # File comparison
    pdf_analysis = report['pdf_analysis']
    md_analysis = report['markdown_analysis']
    
    print("📁 FILE COMPARISON:")
    print(f"  📄 PDF Size: {pdf_analysis['file_size_mb']}MB")
    print(f"  📝 Markdown Size: {md_analysis['file_size_kb']}KB")
    print(f"  🗜️  Compression Ratio: {report['coverage_metrics']['compression_ratio']}:1")
    print()
    
    # Content analysis
    print("📊 CONTENT ANALYSIS:")
    print(f"  📄 PDF Elements: {pdf_analysis['total_elements']:,}")
    print(f"  📝 PDF Characters: {pdf_analysis['total_characters']:,}")
    print(f"  📖 PDF Words: {pdf_analysis['total_words']:,}")
    print(f"  📑 Markdown Characters: {md_analysis['clean_characters']:,}")
    print(f"  📖 Markdown Words: {md_analysis['total_words']:,}")
    print()
    
    # Structure analysis
    structure = md_analysis['structure']
    print("🏗️  STRUCTURE ANALYSIS:")
    print(f"  📖 Main Topics: {structure['main_topics']}")
    print(f"  🎯 Capabilities: {structure['capabilities']}")
    print(f"  ⭐ Features: {structure['features']}")
    print(f"  📝 Descriptions: {structure['descriptions']}")
    print(f"  🔧 Functionality Items: {structure['functionality_items']}")
    print(f"  💡 Use Cases: {structure['use_cases']}")
    print()
    
    # Coverage metrics
    coverage = report['coverage_metrics']
    print("📈 COVERAGE METRICS:")
    print(f"  📊 Content Coverage: {coverage['content_coverage_percent']}%")
    print(f"  🔤 Word Coverage: {coverage['word_coverage_percent']}%")
    print(f"  🔑 Key Terms Coverage: {coverage['key_terms_coverage_percent']}%")
    print(f"  📋 Heading Coverage: {coverage['heading_coverage_percent']}%")
    print(f"  🏗️  Structure Completeness: {coverage['structure_completeness_percent']}%")
    print()
    
    # Quality assessment
    quality = report['quality_assessment']
    print("⭐ QUALITY ASSESSMENT:")
    print(f"  🎯 Overall Score: {quality['overall_score']}/100")
    print(f"  📊 Content Preservation: {quality['content_preservation']}")
    print(f"  🏗️  Structure Quality: {quality['structure_quality']}")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if coverage['content_coverage_percent'] < 70:
        print("  ⚠️  Consider using --enhanced mode for better content extraction")
    if coverage['structure_completeness_percent'] < 100:
        print("  ⚠️  Some structural elements may be missing")
    if len(coverage['missing_key_terms']) > 5:
        print(f"  ⚠️  {len(coverage['missing_key_terms'])} important terms may be missing")
    if quality['overall_score'] >= 80:
        print("  ✅ Excellent conversion quality!")
    elif quality['overall_score'] >= 60:
        print("  ✅ Good conversion quality")
    else:
        print("  ⚠️  Consider reviewing the conversion parameters")
    
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python validator.py <pdf_path> <markdown_path> [report_output_path]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    markdown_path = sys.argv[2]
    report_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        report = generate_validation_report(pdf_path, markdown_path, report_path)
        print_validation_summary(report)
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        sys.exit(1) 