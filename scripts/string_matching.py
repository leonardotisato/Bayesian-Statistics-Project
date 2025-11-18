from fuzzywuzzy import process

def string_matching(dirty_list, clean_list, acceptance_threshold):
    mapping = {}
    unmatched = []
    
    for dirty in dirty_list:
        # find best match
        match, score = process.extractOne(dirty, clean_list)
        
        # check best match score against threshold
        if score >= acceptance_threshold:
            mapping[dirty] = match
        else:
            unmatched.append((dirty, match, score))

    if unmatched:
        print("\n--- Unmatched Names (and their best guess) ---")
        
        print(f"[Note: {len(unmatched)} items were not matched because they were below the {acceptance_threshold}% threshold]")
        for item in unmatched:
            print(f"  - '{item[0]}' -> best guess '{item[1]}' (Score: {item[2]})")

    return mapping