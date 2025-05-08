# CSC 3105 - Data Analytics Mini-Project

Data Analytics of
Line-of-Sight (LOS) & Non-Line-
Sight(NLOS) Wireless Signals

This mini-project serves as an exercise in demonstrating the three stages of Data Analytics, namely Data preparation
(Data cleaning and preprocessing), Data Mining and Data Visualization with result analysis.

## Dataset

The project uses a dataset obtained from [ewine-project](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set)

Please refer to the dataset's documentation for more information on its structure and usage.

For more information on the dataset, please refer to the citation below:

[Klemen Bregar, Andrej Hrovat, Mihael Mohorčič, “NLOS Channel Detection with Multilayer Perceptron in Low-Rate Personal Area Networks for Indoor Localization Accuracy Improvement”. Proceedings of the 8th Jožef Stefan International Postgraduate School Students’ Conference, Ljubljana, Slovenia, May 31-June 1, 2016.](https://www.researchgate.net/publication/308986067_NLOS_Channel_Detection_with_Multilayer_Perceptron_in_Low-Rate_Personal_Area_Networks_for_Indoor_Localization_Accuracy_Improvement)

## Set-up

To set up the python virtual environment, run:

```shell
python -m venv .venv
```

Next, activate said python virtual environment by running the following:

```shell
.venv\Scripts\activate 
```

> [!TIP]
> Helpful advice for doing things better or more easily.
if you have difficulty with running thie activate script on windows, try running ` Set-ExecutionPolicy -ExecutionPolicy Unrestricted ` in PowerShell as an administrator, this will introduct vulnerabilities to your OS so remember to ` Set-ExecutionPolicy -ExecutionPolicy restricted ` when you are done

Install the required packages/dependencies using the included requirements.txt:

```shell
py -m pip install -r requirements.txt
```

To create the kernel using the project's venv, run:

```shell
python -m ipykernel install --user --name csc3105_project --display-name csc3105_project 
```

> [!NOTE]
> If you're in visual studio, select '.venv', if you're in anacondo jupyter notebook web, select 'csc3105_project'

## Authors and acknowledgment

CSC3105 Grp 28 T2 AY2024/25

## License

Creative Commons License

## Project status

Completed 