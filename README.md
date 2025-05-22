# SMART EVALUATOR (Powered by Google Cloud Vision API)

This is a website designed keeping in mind the teachers/students in educational institutions, where the evaluators can upload the image (JPG format) of the handwritten answer script along with the answer key (correct answer, not just key words).

→ Google Cloud Vision API fetches the text from the image  
→ This text is then checked for similarity using an ML model (S-BERT)  
→ Returns a similarity score out of 100
