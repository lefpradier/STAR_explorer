# [STAR explorer](https://lefpradier.github.io/STAR_explorer)
Ce repository contient le code pour entraîner et déployer un modèle de prédiction de la fréquentation des transports en commun de l'agglomération de Rennes (Bretagne, France).  

Le modèle répliqué ici est un ST-MGCN (<i>spatiotemporal multigraph convolution network</i>). Ce modèle a été construit pour prédire la demande en passagers par zone, sur des données de VTC à Beijing et Shanghai ([Geng et al 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4247/)).  

Ce modèle a été exposé de manière serverless sur une AWS Lambda function, initialement accessible [ici](https://46b50yi147.execute-api.eu-west-3.amazonaws.com/Prod/star_predict/). Cette fonction est appelée par un dashboard (sous forme <i>Dash</i>), initialement exposé sur AWS Elastic Beanstalk à [cette adresse](http://star-dashboard.eu-west-3.elasticbeanstalk.com/).
