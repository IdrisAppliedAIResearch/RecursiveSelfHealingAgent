You are extracting findings and results from a neuroscience abstract.

Return ONLY a valid JSON object with this exact structure:
{"claims": ["finding 1", "finding 2", ...]}

Extract sentences that report WHAT the study found. Examples of findings:
- "Activation in the prefrontal cortex was significantly increased"
- "Patients showed reduced connectivity compared to controls"
- "Performance improved following treatment"
- "No significant difference was found"

Do NOT extract sentences about methods, participants, equipment, or study aims.

If no findings are present, return {"claims": []}.
