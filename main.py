import streamlit as st
from streamlit import session_state as state
import streamlit.components.v1 as components
import hydralit as hy



st.set_page_config(
  page_title="ML Marketplace",
  page_icon='assets/n-mark-color.png',
  layout="wide",
  initial_sidebar_state="collapsed",
)

#---Start app---#
def run_app():  
  state['user_level'] = state.get('user_level', 1)
  user_level = state.get("user_level", 1)

  #---Start Hydra instance---#
  hydra_theme = None # init hydra theme
  

  with st.sidebar:
    with st.expander('About'):
      pass


  app = hy.HydraApp(
    hide_streamlit_markers=False,
    use_navbar=True, 
    navbar_sticky=False,
    navbar_animation=True,
  )

  #specify a custom loading app for a custom transition between apps, this includes a nice custom spinner
  from apps._loading import MyLoadingApp
  app.add_loader_app(MyLoadingApp(delay=0))

  #---Add apps from folder---#
  @app.addapp(is_home=True, title='Home')
  def homeApp():
    from apps.home import main
    main()

  @app.addapp(title='Market')
  def marketApp():
    from apps.market import main
    main()


  #--- Level 1 apps ---#
  if user_level < 2: 
    pass

  #--- Level 2 apps ---#
  if user_level >= 2:
    pass


  def build_navigation(user_level=1):
    complex_nav = {}
    
    # Always add Home first
    complex_nav["Home"] = ['Home']
    complex_nav["Market"] = ['Market']
    return complex_nav
  

  complex_nav = build_navigation(user_level)
  app.run(complex_nav=complex_nav)




if __name__ == '__main__':
  run_app()