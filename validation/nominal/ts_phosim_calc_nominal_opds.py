import os
import subprocess
import numpy as np
import yaml
from lsst.ts.wep.utility import FilterType

from lsst.ts.phosim.telescope.TeleFacade import TeleFacade
from lsst.ts.phosim.OpdMetrology import OpdMetrology
from lsst.ts.phosim.utils.Utility import getConfigDir, getPhoSimPath, getAoclcOutputPath


def main(phosimDir):
    # Settings
    outputDir = getAoclcOutputPath()
    outputImgDir = os.path.join(outputDir, "img")
    os.makedirs(outputImgDir, exist_ok=True)

    configDir = getConfigDir()
    cmdSettingFile = os.path.join(configDir, "cmdFile", "opdDefault.cmd")
    instSettingFile = os.path.join(configDir, "instFile", "opdDefault.inst")

    # Declare the opd metrology and add the interested field points
    metr = OpdMetrology()
    metr.setWgtAndFieldXyOfGQ("lsst")
    wfsXY = metr.getDefaultLsstWfsGQ()
    metr.addFieldXYbyDeg(*wfsXY)

    # Write out the field points
    with open("fieldXY.yaml", "w") as f:
        yaml.dump({
            'x': metr.fieldX.tolist(),
            'y': metr.fieldY.tolist()
            }, f
        )

    # Set the Telescope facade class
    tele = TeleFacade()
    tele.setPhoSimDir(phosimDir)

    obsId = 9006050
    filterType = FilterType.REF
    tele.setSurveyParam(obsId=obsId, filterType=filterType)

    # Update the telescope degree of freedom with sepecific camera dx
    dofInUm = np.zeros(50)
    tele.accDofInUm(dofInUm)

    # Write the physical command file
    cmdFilePath = tele.writeCmdFile(
        outputDir, cmdSettingFile=cmdSettingFile, cmdFileName="opd.cmd"
    )

    # Write the instance file
    instFilePath = tele.writeOpdInstFile(
        outputDir, metr, instSettingFile=instSettingFile, instFileName="opd.inst"
    )

    # Get the argument to run the PhoSim
    logFilePath = None #os.path.join(outputImgDir, "opdPhoSim.log")
    argString = tele.getPhoSimArgs(
        instFilePath,
        extraCommandFile=cmdFilePath,
        numPro=2,
        outputDir=outputImgDir,
        e2ADC=0,
        logFilePath=logFilePath,
    )
    # Run PhoSim
    tele.runPhoSim(argString)

    os.makedirs(os.path.join(outputDir, "..", "phosim"), exist_ok=True)
    for iopd in range(35):
        ifn = f"opd_9006050_{iopd}.fits.gz"
        ofn = ifn.replace("9006050", f"nominal_field")
        cmd = [
            "cp",
            os.path.join(outputImgDir, ifn),
            os.path.join(outputDir, "..", "phosim", ofn)
        ]
        subprocess.call(" ".join(cmd), shell=True)


if __name__ == "__main__":

    phosimDir = getPhoSimPath()
    main(phosimDir)
