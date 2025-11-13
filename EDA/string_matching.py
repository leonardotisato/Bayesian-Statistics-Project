from fuzzywuzzy import process

def string_matching(dirty_list, clean_list, acceptance_threshold):
    mapping = {}
    unmatched = []
    
    for dirty in dirty_list:
        # Find the single best match from the clean list
        match, score = process.extractOne(dirty, clean_list)
        
        # If the score is high enough, add it to our mapping
        if score >= acceptance_threshold:
            mapping[dirty] = match
        else:
            unmatched.append((dirty, match, score))

    if unmatched:
        print("\n--- Unmatched Names (and their best guess) ---")
        # --- FIX 1 ---
        print(f"[Note: {len(unmatched)} items were not matched because they were below the {acceptance_threshold}% threshold]")
        for item in unmatched:
            print(f"  - '{item[0]}' -> best guess '{item[1]}' (Score: {item[2]})")

    # --- 3. Apply the Mapping ---

    # --- FIX 2 (The Critical One) ---
    # Create a new column 'clean_province' in the work_rates DataFrame
    return mapping