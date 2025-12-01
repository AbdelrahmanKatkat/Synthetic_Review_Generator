import json
import re
from pathlib import Path

def parse_reviews_from_markdown(md_file):
    """Parse reviews from markdown file and extract rating and review text."""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    reviews = []
    review_id = 1
    
    # Split by review sections - look for rating patterns
    lines = content.split('\n')
    
    current_review = {}
    current_text_parts = []
    in_review_section = False
    
    for i, line in enumerate(lines):
        # Look for rating pattern like "4.5/5" or "5/5"
        rating_match = re.match(r'^(\d+(?:\.\d+)?)/5$', line.strip())
        
        if rating_match:
            # Save previous review if exists
            if current_review and current_text_parts:
                current_review['text'] = ' '.join(current_text_parts).strip()
                if current_review.get('rating') and current_review.get('text'):
                    reviews.append(current_review.copy())
            
            # Start new review
            rating_val = float(rating_match.group(1))
            current_review = {'rating': int(round(rating_val))}
            current_text_parts = []
            in_review_section = True
            continue
        
        # Look for review title (before rating usually)
        if line.startswith('"') and line.endswith('"') and not rating_match:
            # This is likely a review title, include it in text
            if in_review_section:
                current_text_parts.append(line.strip('"'))
        
        # Look for "What do you like best" answers
        elif 'What do you like best' in line or 'What do you dislike' in line:
            # The answer is usually in the next line
            if i + 1 < len(lines):
                answer = lines[i + 1].strip()
                if answer and not answer.startswith('Review collected') and not answer.startswith('What do you'):
                    current_text_parts.append(answer)
    
    # Add last review
    if current_review and current_text_parts:
        current_review['text'] = ' '.join(current_text_parts).strip()
        if current_review.get('rating') and current_review.get('text'):
            reviews.append(current_review)
    
    # Add IDs
    for idx, review in enumerate(reviews, 1):
        review_id = f"real_{idx:05d}"
        review['id'] = review_id
    
    return reviews

def main():
    # Input and output files
    input_file = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/collected_dataset.md')
    output_file = Path('e:/On Going Projects/Capital Growth/Job Growth/Easygenerator/Synthetic_Reviews/Synthetic_Review_Generator/datasets_comparison/real_dataset.jsonl')
    
    # Parse reviews
    reviews = parse_reviews_from_markdown(input_file)
    
    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for review in reviews:
            # Only include id, rating, and text
            output_review = {
                'id': review['id'],
                'rating': review['rating'],
                'text': review['text']
            }
            f.write(json.dumps(output_review) + '\n')
    
    print(f"Converted {len(reviews)} reviews")
    print(f"Output saved to: {output_file}")

if __name__ == '__main__':
    main()
