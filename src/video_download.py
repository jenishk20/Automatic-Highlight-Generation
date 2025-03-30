from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="../data")

mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv", "Labels-v2.json"], split=["train","valid","test","challenge"])