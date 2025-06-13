I would like to create some evals to measure workflow performance both on the individual subtasks and on the end-to-end workflow. We should create the most naive possible version of the end-to-end workflow and start with that. Then I'll see if I can improve on that. To measure performance, I can probably just use cosine similarity to a human-produced result.

The steps, as I see them, roughly in order of priority/practicality:

1. Create, for a few PDF pages, my own ideal human-produced example and a simple eval scoring script.
2. Try giving Google Gemini a few PDF pages with a prompt to see what it can do out of the box. Try with both the actual PDF pages extracted from file and then with images of the pages.
3. Run a PDF through a few commercial and open-source tools to see what outputs we get.
4. Try using an LLM to draw bounding boxes around both text and images with an array to indicate reading order, and then mechanically extract. (Section hierarchy a challenge here; can we enrich with hierarchical heading extraction?)
5. Try mechanical text extraction with markdown and see if maybe we can de mechanical image extraction as well, even though the images are SVGs and their boundaries are therefore hard to detect. (Reading order and section hierarchy a challenge here; can we guess them mechanically?)
6. Try mechanical text extraction plus VLM image extraction. (Reading order and section hierarchy a challenge here; can we guide with an LLM?)

There's a bunch of non-visible text here that we need to strip out. Here is Gemini's analysis of that:

```markdown
Yes, your observation is correct. The text is invisible because it is positioned behind a full-page image.

Here is a breakdown of why:

1.  **Text Block:** The text "West Bank and Gaza Country Climate and Development Report" is located in `blocks[0]`. Its bounding box is `[253.75, 738.42, 512.63, 751.29]`.

2.  **Image Block:** There is a full-page background image defined in `blocks[2]`. Its bounding box is `[0.0, 0.0, 612.0, 792.0]`, which covers the entire page and therefore completely overlaps the area where the text is located.

3.  **Stacking Order:** In this JSON structure, blocks are typically rendered in the order they appear. Since the text block `blocks[0]` comes before the full-page image `blocks[2]`, the image is drawn on top of the text, obscuring it from view.

Additionally, the text color is `2301728` (a very dark gray, almost black). The footer area of the document is also dark, so even if the text were visible, it would have extremely low contrast against the background.
```

We also should handle cases where the alpha channel is 0 and where the text is positioned outside the page area.

For some reason, the `extract_text.py` script does not appear to extract page numbers in the footer. Possibly the information is stored separately from the page object.

Things I want to prompt for and test:
1. Logical page numbers are correctly mapped to PDF page numbers.
2. We have FRONT_MATTER and BODY_MATTER at the top level, correctly sequenced and paginated (though assignment of page 12 to either or neither section is permissible).
3. Invisible text is correctly identified and removed.
4. No visible text is removed (unless it's redundant, part of a header/footer, or trailing whitespace, in which case including it is optional).
5. All array items match the schema.

I'm a bit torn on how to handle markdown formatting, particularly bold, italic, and header text. I *think* it might be optimal to leave headings unformatted, and then use the hierarchy of document components at runtime to add the appropriate formatting. That way we don't make mistakes in adding such formatting during ingestion, and if we edit the document component graph, we don't also have to edit the markdown. We could perhaps use a Postgres function to add the appropriate formatting to markdown headings based on the component hierarchy. We can then hardcode the markdown formatting for any other bold or italic text in the markdown that isn't just part of a heading style. This gets a bit dicey in that it adds a lot of complexity to the ingestion process and the component hierarchy, but it makes things easier at query time.