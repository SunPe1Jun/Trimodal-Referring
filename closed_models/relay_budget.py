"""Shared USD wallet guard for all three full baseline jobs.

The relay exposes several meters: ``cost`` is the nominal pre-discount
amount, ``actual_cost`` is the discounted amount, and ``account_cost`` is an
account-level meter that is not the user's wallet debit. The user authorized
a USD 300 wallet spend cap, so the guard must not use ``account_cost``.
"""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from urllib.parse import urlparse
import requests

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'closed_models/outputs/full_budget_state.json'
_ORIGINAL_RESOLVER=socket.getaddrinfo
_FALLBACKS={}
_INSTALLED=False


def install_relay_dns_fallback(config):
    """Try normal DNS first; fallback only for the configured relay, retaining TLS checks."""
    global _INSTALLED
    host=urlparse(config['base_url']).hostname
    if config.get('dns_fallback_ip'):
        _FALLBACKS[host]=config['dns_fallback_ip']
    if _INSTALLED:return
    def resolve(host,port,*args,**kwargs):
        try:return _ORIGINAL_RESOLVER(host,port,*args,**kwargs)
        except socket.gaierror:
            if host in _FALLBACKS:
                return _ORIGINAL_RESOLVER(_FALLBACKS[host],port,*args,**kwargs)
            raise
    socket.getaddrinfo=resolve
    _INSTALLED=True


def snapshot(config):
    install_relay_dns_fallback(config)
    secret=config.get('api_key') or os.environ.get(config.get('api_key_env','TRIMODAL_OPENAI_API_KEY'))
    response=requests.get(config['base_url'].rstrip('/')+'/usage',
        headers={'Authorization':'Bearer '+secret},timeout=25,allow_redirects=False)
    if response.status_code!=200:raise RuntimeError('Wallet read HTTP '+str(response.status_code))
    result=response.json()
    if result.get('unit')!='USD' or not isinstance(result.get('balance'),(int,float)):
        raise ValueError('Wallet USD balance unavailable')
    return result


def meters(usage):
    return {key:sum(float(r.get(key,0)) for r in usage.get('model_stats',[]))
            for key in ['actual_cost','account_cost','cost']}


def initialize(config,cap=300.,reserve=10.):
    data=snapshot(config)
    STATE.parent.mkdir(parents=True,exist_ok=True)
    state={'approved_run_cap_usd':cap,'minimum_wallet_balance_usd':reserve,
           'initial_balance_usd':data['balance'],'initial_usage':data,
           'last_balance_usd':data['balance'],'observed_balance_debits_usd':0.,
           'status':'AUTHORIZED_FULL_BASELINES','scope':'three legacy full baselines, 4000 each; no new ablations',
           'user_reported_multiplier':.1,'extra_funding_authorized_usd':300.,
           'meter_rule':'max(actual_cost delta, observed wallet debits); account_cost and nominal cost excluded'}
    STATE.write_text(json.dumps(state,indent=2)+'\n')
    return state


def check(config,request_count):
    """Reserve budget for one small paid batch; lock serializes cross-setting checks."""
    lock=STATE.with_suffix('.lock')
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        state=json.loads(STATE.read_text())
        data=snapshot(config)
        current=meters(data);initial=meters(state['initial_usage'])
        state['observed_balance_debits_usd']+=max(0,state['last_balance_usd']-data['balance'])
        state['last_balance_usd']=data['balance']
        # ``account_cost`` is a relay/account meter in pre-discount units and
        # can exceed the user's wallet debit by 10x. The authorized cap is in
        # wallet USD, so use discounted actual cost and observed balance debits.
        actual_delta=max(0., current['actual_cost']-initial['actual_cost'])
        debit=max(state['observed_balance_debits_usd'], actual_delta, 0.)
        # Wallet-space reservation, intentionally conservative; never presented as price.
        reserve=float(config.get('budget_reserve_per_request_usd',.25))*request_count
        outstanding={k:v for k,v in state.get('batch_reservations',{}).items()
                     if Path('/proc',k).exists()}
        others=sum(v['amount_usd'] for k,v in outstanding.items() if k!=str(os.getpid()))
        enough_funds=data['balance']>=state['minimum_wallet_balance_usd']+reserve+others
        enough_cap=debit+reserve+others<=state['approved_run_cap_usd']
        if enough_funds and enough_cap:
            outstanding[str(os.getpid())]={'amount_usd':reserve,'requests_upper_bound':request_count}
        state.update(batch_reservations=outstanding,latest_usage=data,conservative_spend_since_resume_usd=debit,
                     latest_nominal_cost_delta=current['cost']-initial['cost'],
                     status='RUNNING' if enough_funds and enough_cap else 'WAITING_TOPUP' if enough_cap else 'CAP_REACHED')
        STATE.write_text(json.dumps(state,indent=2)+'\n')
        return state['status'],data['balance'],debit


def release():
    with STATE.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        state=json.loads(STATE.read_text())
        state.setdefault('batch_reservations',{}).pop(str(os.getpid()),None)
        STATE.write_text(json.dumps(state,indent=2)+'\n')
