import os
import re

def main():
    filepath = '/Users/donthireddy/code/ai-course/docs/complete_illustrated_coursebook.md'
    workspace_dir = '/Users/donthireddy/code/ai-course'
    file_dir = os.path.dirname(filepath)
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} does not exist.")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all Markdown images: ![alt](path)
    md_images = re.findall(r'!\[.*?\]\((.*?)\)', content)
    
    # Find all HTML images: <img src="path" ...> or similar
    html_images = re.findall(r'<img\s+[^>]*src=["\']([^"\']+)["\']', content)
    
    all_images = md_images + html_images
    unique_images = list(set(all_images))
    
    print(f"Total image references found: {len(all_images)}")
    print(f"Unique image references found: {len(unique_images)}")
    
    missing_images = []
    found_images = 0
    
    for img_path in all_images:
        # Resolve path relative to workspace or absolute
        if img_path.startswith('/'):
            full_path = img_path
        else:
            full_path = os.path.normpath(os.path.join(file_dir, img_path))
            
        if not os.path.exists(full_path):
            missing_images.append((img_path, full_path))
        else:
            found_images += 1
            
    print(f"Valid images found: {found_images}")
    if missing_images:
        print(f"\nWARNING: Found {len(missing_images)} missing/broken image references:")
        for idx, (rel, full) in enumerate(missing_images, 1):
            print(f"  {idx}. Reference: '{rel}' -> Resolved: '{full}'")
    else:
        print("\nSUCCESS: All image references resolved successfully!")

    # Find all regular markdown links: [text](path) - make sure it doesn't match ![text](path)
    # We do a negative lookbehind for '!'
    regular_links = re.findall(r'(?<!\!)\[.*?\]\((.*?)\)', content)
    unique_links = list(set(regular_links))
    print(f"\nTotal regular links found: {len(regular_links)}")
    print(f"Unique regular links found: {len(unique_links)}")

    missing_links = []
    external_links_count = 0
    for link_path in unique_links:
        # Skip web links
        if link_path.startswith(('http://', 'https://', '#', 'mailto:')):
            external_links_count += 1
            continue
        # Also skip file:// if needed or parse it
        path_to_check = link_path
        if link_path.startswith('file:///'):
            path_to_check = link_path[7:]
        
        # Split anchor if present
        if '#' in path_to_check:
            path_to_check = path_to_check.split('#')[0]
            
        if not path_to_check:
            # Anchor only link within same file
            continue

        if path_to_check.startswith('/'):
            full_path = path_to_check
        else:
            full_path = os.path.normpath(os.path.join(file_dir, path_to_check))

        if not os.path.exists(full_path):
            missing_links.append((link_path, full_path))

    print(f"External/anchor links skipped: {external_links_count}")
    if missing_links:
        print(f"WARNING: Found {len(missing_links)} missing/broken local links:")
        for idx, (rel, full) in enumerate(missing_links, 1):
            print(f"  {idx}. Reference: '{rel}' -> Resolved: '{full}'")
    else:
        print("SUCCESS: All local links resolved successfully!")

if __name__ == '__main__':
    main()
