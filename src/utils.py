

def save_as_html(preds, labels, inputs, path="predictions.html"):

    with open(path, "w", encoding='utf-8') as f:
        f.write("<html><body><h1>predictions</h1>")
        for i, (inp, lbl, pred) in enumerate(zip(inputs, labels, preds)):
            f.write(f"<b>Source:</b> {' '.join(inp)}<br>")
            f.write(f"<b>Target:</b>  {lbl}<br>")
            f.write(f"<b>Pred:</b> {pred}<br><hr>")
        f.write("</body></html>")