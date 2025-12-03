1) runRegression 

Run all test cases for compliance test, test vector generation and 
performance test with pre-defined test configuration.

- Usage:      runRegression(testSet, channelSet, caseSet) 
- testSet:    Compliance, TestVector, Performance, allSets
- channelSet: ssb, pdcch, pdsch, csirs, dlmix, allDL, prach, pucch, pusch, srs, ulmix, allUL, allChannels 
- caseSet:    compact, full, selected (choose one set only)

Examples:    

- runRegression({'Compliance', 'TestVector'}, {'allDL', 'prach', 'srs'}, 'full')
- runRegression() = runRegression({'allSets'}, {'allChannels'}, 'compact')

2) runSim

Run a single test case with user defined test configuration through 
input yaml file.  

- Run genCfgTemplate to generate a configuration template "cfg_template.yaml". 
- Edit the template file with specific configuration and save it to another yaml file name.
- Run runSim(cfgFileName, tvFileName). For example, runSim('pdsch_f14_cfg.yaml', 'pdsch_f14');

runSim() will run with the default configuration.