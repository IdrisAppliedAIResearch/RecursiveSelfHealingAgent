Extract scientific claims (findings, results) from this neuroscience abstract.

Return ONLY valid JSON in this format:
{"claims": ["claim 1", "claim 2"]}

WHAT TO INCLUDE (extract these as claims):
- Statistical findings: "X was significantly greater than Y (p < 0.05)"
- Brain activations/deactivations: "Increased activation in [region] during [condition]"
- Connectivity changes: "Enhanced functional connectivity between A and B"
- Group differences: "Patients showed reduced activation compared to controls"
- Behavioral outcomes: "Performance improved following intervention"
- Correlations: "Activation in region X correlated with behavioral measure Y"
- Predictive relationships: "Baseline activity predicted treatment response"
- Null findings: "No significant difference was found between groups"
- Mediation/moderation: "Effect was mediated by variable Z"

WHAT TO EXCLUDE (do NOT extract these):
- Methods and procedures: "We used fMRI to scan participants"
- Participant details: "30 healthy volunteers aged 24-35"
- Equipment specifications: "Images acquired at 3T with TR=2000ms"
- Study aims and objectives: "We aimed to investigate the neural basis of..."
- Background and prior work: "Previous studies have shown that..."
- Statistical methods: "Data were analyzed using a general linear model"
- Preprocessing steps: "Images were preprocessed using SPM12"
- Task descriptions: "Participants performed an n-back working memory task"
- Sample sizes and demographics: "N=45, mean age 28 (SD=5)"
- Inclusion/exclusion criteria: "Participants were right-handed with normal vision"

CLAIM FORMAT:
- Each claim must be a complete sentence describing a specific finding
- Include relevant statistical details (p-values, effect sizes) when present
- Each claim should be independent and self-contained
- Do not combine multiple distinct findings into one claim
- If the abstract contains no findings, return {"claims": []}

IMPORTANT: Only extract sentences that report what the study FOUND, not how the study was conducted.