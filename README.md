# hw4-236781
Deep learning research project code.

i. Differences in structure to original code:
  1. src/attacks/apgd implements APGD attack
  2. src/experiments implements experiments helpers classes as a package
  3. src/run_experiment.py implements the tool to run the experiments in the report
  4. src/run_experiment_evaluation.py implements a tool to evaluate the experiments using the const attack
  5. src/plot.py implements plotting and saving figures and other important data from the results of the experiments
 After using plot.py:
  6. src/experiment_results/  A folder that contains plots and additional results of experiments


Where to put data and how to process it:
  1. Process the data using the original coded provided with the assignment. (We succeeded once to make that work with our code and with train\val\test datasets but lost that code)
  2. Copy data/ folder (including processed data) to src/

ii. How to reproduce our results:
  1. Include data and processed data under src/data/ as described above
  2. cd into src/ (this should always be your wd)
  3. Activate project environment
  4. run "python3 run_experiment.py --attack [attack] --optimizer [optimizer]", if you want to run on lambda servers with slurm add "--slurm" to the end of the command.
  "--attack" can be apgd or pgd.
  when attack=apgd optimizer can be "I_FGSM" or "MI_FGSM"
  when attack=pgd optimizer can be "MI_FGSM", "AB_FGSM", "M_AB_FGSM". "MI_FGSM" gives results for I-FGSM experiment in the report, that is when mu=0
  5. run "python3 run_experiment_evaluation.py --attack [attack] --optimizer [optimizer]", if you want to run on lambda servers with slurm add "--slurm" to the end of the command.
  "--attack" and "--optimizer" options are exactly the same as in step 4. except here for attack=pgd and for the I-FGSM as in the report specify "--optimizer=I_FGSM" as well.
  6. run "python3 run_experiment.py --attack [attack] --optimizers [optimizer [optimizer ...]]"
  same as step 5, except you can specify multiple optimizers at once when using attack=pgd. you can also use one at a time.

  Note: You can run one experiment at a time by doing steps 4 5 6 only for one attack and optimizer (note that for I-FGSM in 5 and 6 you must run MI-FGSM in 4).

  Now, you can find the relevant plots under the src/experiments_results folders, under subfolders for each attack and optimizer configuration.
  Although the names may be inaccurate, the inplace.csv corresponds to the M_RMS table in section 4 of the report and out_of_place.csv corresponds to the M_VO table.
 
 What follows is a typical command to run an attack. This line explains which attacks can have which optimizers, and it provides options for hyper paramters of an attack. Note that alpha in case of apgd specifies the initial alpha as in the report.
 You can choose which data set you want for evaluation using --val_set_idx and which for test using --test_set_idx. Those should be different always.
 --save_report_run redirects the output to the results folder of the attack
python3 run_attacks.py --seed 42 --model-name tartanvo_1914.pkl --test-dir "VO_adv_project_train_dataset_8_frames"  --max_traj_len 8 --batch-size 1 --worker-num 1 --save_csv --attack_k 20  --preprocessed_data --save_report_run --save_best_pert --attack [pgd, apgd] --attack_optimizer [{apgd: MI_FGSM, I_FGSM}, {pgd: MI_FGSM, I_FGSM, AB_FGSM, M_AB_FGSM}] [MI_FGSM: --mu float] [--beta_1 float] [--beta_1 float] [--val_set_idx int] [--test_set_idx int] [--sparsity_ratio_threshold float]
