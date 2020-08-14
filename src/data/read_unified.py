verb = False
verbosed = []

# NOTE: remove('') removes only one entry

def get_emb_representation(word: str, vocab: set) -> str:
    if word == "":
        return ""
    elif word in vocab:
        return word
    elif word.lower() in vocab:
        if verb and word not in verbosed:
            print("Taking label lower-case:", word, "->", word.lower())
            verbosed.append(word)
        return word.lower()
    else:
        w = word.replace(" ", "")
        if w in vocab:
            if verb and word not in verbosed: 
                print("Taking label without spaces:", word, "->", w)
                verbosed.append(word)
            return w
        elif w.lower() in vocab:
            if verb and word not in verbosed:
                print("Taking label without spaces and lower-case:", word, "->", w.lower())
                verbosed.append(word)
            return w.lower()
        else:
            res = []
            for term in word.split(" "):
                if term in vocab:
                    res.append(term)
                elif term.lower() in vocab:
                    res.append(term.lower())
                else:
                    if verb and word not in verbosed:
                        print("Removed word", term, "from label", word, "because it is not contained in the wordembeddings.")
                        verbosed.append(word)
            lbl = " ".join(res)
            if verb and lbl != word and word not in verbosed:
                print("Taking label as following:", word, "->", lbl)
                verbosed.append(word)
            return lbl

def read_unified(data: dict, threshold: float, vocab: set, optimize: bool):
    assert "columns" in data.keys()
    assert "score_columns" in data.keys()
    assert "predictions" in data.keys()
    assert "unification_method" in data.keys()

    cols = data["columns"]
    score_cols = data["score_columns"]
    predictions = data["predictions"]
    unify_method = data["unification_method"]
    
    rows_labels = []
    rows_scores = []
    rows_img_id = []
    has_scores = score_cols != []

    for img_id, rows in predictions.items():
        row_vals = []
        row_scores = []

        for col_i, col in enumerate(cols):
            col_val = rows[col]
            col_val = unify_method(col_val)
            
            if optimize:
                col_val = get_emb_representation(col_val, vocab)
                if col_val != "" and col_val not in vocab:
                    for token in col_val.split(' '):
                        assert token in vocab
            
            # ignore empty entries
            if not col_val:
                continue

            col_score = 1
            if has_scores:
                score_col = score_cols[col_i]
                col_score = float(rows[score_col])
                if col_score < threshold:
                    continue
            row_vals.append(col_val)
            row_scores.append(col_score)
        
        assert len(row_vals) == len(row_scores)
        rows_labels.append(row_vals)
        rows_scores.append(row_scores)
        rows_img_id.append(img_id)

    return (rows_img_id, rows_labels, rows_scores)
