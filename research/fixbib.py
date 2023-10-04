
def do_one_paper(lines, i):
    
    header = lines[i]
    i += 1
    
    link = ""
    
    bib_flag = False
    bib = []
    
    new_paper_data = []
    
    
    while i < len(lines) and "###" not in lines[i]:
        line = lines[i].strip()
        if '- [link]' in line:
            link = line[2:]
        
        if "```" in line and bib_flag:
            bib.append(line)
            break
        
        if "```" in line and not bib_flag:
            bib_flag = True
        
        if bib_flag:
            bib.append(line)
        
            
        if not bib_flag and not '- [link]' in line:
            new_paper_data.append(line)
        
        i += 1
    
    if link == "":
        return '\n'.join([header] + new_paper_data), i-1
    

    bib[0] += "bibtex"
    
    print(repr(bib[0]))
    
    header_reference = header.lower().replace("###", "").strip().replace(" ", "-")    
    new_header = header + " (" + link + "/[citation](./bibliography.md#" + header_reference + "))"
    
    with open('bibliography.md', 'a') as f:
        
        text = [header] + [""] + bib
        
        print('\n'.join(text), file=f)
    
    
    new_paper_data = [new_header] + new_paper_data
    
    text = '\n'.join(new_paper_data)
    
    return text, i
    


with open('papers.md', 'r') as f:
    lines = f.readlines()
    
    i = 0 
    
    papers = []
    
    while i < len(lines):
        
        text = lines[i]
        if '###' in lines[i]:
            text, i = do_one_paper(lines, i)
            papers.append(text)
            print(text)
            
        with open('papers2.md', 'a') as f:
            print(text.strip(), file=f)

        i += 1
