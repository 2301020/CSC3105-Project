# CSC3105-Project

` python -m venv .venv `

` .venv\Scripts\activate `

if you have difficulty with running thie activate script on windows, try running ` Set-ExecutionPolicy -ExecutionPolicy Unrestricted ` in PowerShell as an administrator, this will introduct vulnearbilities to your OS so remember to ` Set-ExecutionPolicy -ExecutionPolicy restricted ` when you are done

` py -m pip install -r requirements.txt `

` python -m ipykernel install --user --name csc3105_project --display-name csc3105_project ` to create the kernel using the project's venv

if you're in visual studio, select '.venv', if you're in anacondo jupyter notebook web, select 'csc3105_project'