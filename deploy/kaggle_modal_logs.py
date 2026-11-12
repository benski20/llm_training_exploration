import argparse,os,subprocess,sys
ap=argparse.ArgumentParser();ap.add_argument("--id");ap.add_argument("--secret");a=ap.parse_args()
subprocess.run([sys.executable,"-m","pip","install","-q","modal==1.6.3"],check=True)
env=dict(os.environ,MODAL_TOKEN_ID=a.id,MODAL_TOKEN_SECRET=a.secret)
r=subprocess.run("timeout 45 modal app logs verichat",shell=True,env=env,capture_output=True,text=True)
out=(r.stdout+r.stderr).replace(a.secret,"***").replace(a.id,"***")
print(out[-12000:])
