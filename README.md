# Software Documentation 
### Requirements
- Rocket League (Epic Games or Steam Version)
- BakkesMod
- RLBotGUI
- Python Version betwen 3.7 and 3.9
- RLGym
- Stable-baselines3
- RLGym-tools


### Installation

Rocket League can be downloaded via the Steam or Epic Games. It is important to note that to run multiple instances of the game for training, the Epic Games version of Rocket League must be used.

BakkesMod can be downloaded from bakkesmod.com.

The RLBotGUI can be downloaded from RLBot.org.

The version of Python you are running must be between 3.7 and 3.9 -- as RLGym is not compatible with any versions outside of this range. 

The RLGym plugin for BakkesMod can be downloaded by pip installing it via a terminal/command prompt: 

`pip install rlgym`

StableBaselines3 can also be pip installed via:

 `pip install stable-baselines3[extra]`

Finally RLGym-tools can be pip installed via:

`pip install rlgym-tools`



### Running the Code (Training)

To run this code ensure that all packages, libraries, APIs, etc. that are mentioned above are installed properly. Make sure that the RLGym plugin is turned on within the BakkesMod interface, under the "plugins" tab. 

To begin training, open and run the tutorial_bot.py file. Check to make sure the number of instances of Rocket League isn't too high for your computer to handle. Next, ensure that BakkesMod is running so that it can inject into Rocket League as the game is launching. After hitting run, Rocket League should automatically open up and take you to an exhibition match. The in-game timer will countdown and then the model will begin training. 

Additionally, it is important to clear out the "logs," "models," and "mmr_models" before you start training -- if you would like to start with a brand new bot. 

### Uploading Model to RLBot for Evaluation

If you would like to upload your own model, insert your <model_name>.zip file into the RLBot Config directory. Next, open the agent class and identify this section of code:

`self.actor = PPO.load(str(_path) + '/exit_save.zip', device='cpu', custom_objects=custom_objects)`

Change `exit_save.zip` to your <model_name>.zip.  Otherwise, the model being uploaded will be my most recent model that is included in the config directory. Finally, within the RLBotGUI, delete any bots currently under the blue team and orange team. Then, hit the add button near the top left. This will allow you to navigate to the RLBot Config directory and upload the bot.cfg file. After uploading the bot.cfg file, a new bot should appear in the list of all of the other bots. Its name should appear as Oswald. Setup the teams however you would like and then hit "Launch Rocket League and Start Match." Make sure that any instances of Rocket League are closed before doing this, or else you will get an error. 
