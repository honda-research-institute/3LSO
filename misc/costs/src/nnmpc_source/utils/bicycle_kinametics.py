from system_plant import SystemPlant
class BycicleKinematics(SystemPlant):
    def __init__(self) -> None:
        pass

    def update(self, state, action) -> list:
        """Update the state with the actions

        Args:
            state (_type_): _description_
            action (_type_): _description_

        Returns:
            list: next state
        """