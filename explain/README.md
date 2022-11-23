### Generating Explanations

- To compute text block importance with MSP, adjust the MSP parameters in `params.yml`, then run `python explain_with_msp.py`.  
- To compute text block importance with SOC, adjust the SOC parameters in `params.yml`, then run `python explain_with_soc.py`.

Be sure to check out and adjust the input data path, trained classifier path, and LM path (in the case of SOC), as well as the parameters for the explainability algorithms before running each script.  MSP can generate sentence-level explanations as well as explanations on fixed-length blocks of text.  MSP can also be run in Data Parallel mode to speed up inference.
