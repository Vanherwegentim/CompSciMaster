Nicolas Scheers
Brian Verbanck


Setup:
Java SDK 17
Google Cloud SDK
https://cloud.google.com/sdk/docs/install-sdk
https://cloud.google.com/appengine/docs/standard/setting-up-environment?tab=python

Firebase
https://firebase.google.com/docs/cli#install-cli-windows

run firebase:
firebase emulators:start --project demo-distributed-systems-kul

Kill process 8081:
netstat -ano | findstr :<PORT>
taskkill /PID <PID> /F

