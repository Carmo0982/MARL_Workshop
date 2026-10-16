from __future__ import annotations
import argparse

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scenario',default='stable_day',
        choices=['stable_day','trending_day','volatile_day','low_liquidity','stress_test','rmsc04'])
    p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    try:
        from abides_markets.config_system import SimulationBuilder
        from abides_markets.simulation import run_simulation
    except ImportError as e:
        raise SystemExit('ABIDES-NG is not installed. Run SETUP_ONCE.bat first.') from e
    config=(SimulationBuilder().apply_template(a.scenario).ticker('ABM').seed(a.seed).build())
    result=run_simulation(config)
    print(result.summary())
    print('\nTop five agents by PnL:')
    for agent in sorted(result.agents,key=lambda x:x.pnl_cents,reverse=True)[:5]:
        print(f"{agent.agent_type:25s} PnL=${agent.pnl_cents/100:,.2f}")

if __name__=='__main__': main()
