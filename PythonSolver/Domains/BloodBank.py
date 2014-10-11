import numpy as np

from Domain import Domain

class BloodBank(Domain):

    def __init__(self,Network,alpha=2):
        # unpack network
        # will need to update all the functions to use self.variable
        self.Dim = Network.Dim # figure out later
        self.alpha = alpha

    def F(self,Data):

        # pack data

        F =  FX_dX(x,\
        u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
        gam_1C,gam_CB,gam_BP,gam_PS,gam_SD,gam_DR,\
        alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
        alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
        mu,\
        chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
        chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
        zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
        rhat,\
        pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
        pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR,\
        ubar_1C,ubar_CB,ubar_BP,ubar_PS,ubar_SD,ubar_DR,\
        ind_C___C,\
        ind_CB__C,ind_CB__B,\
        ind_CBD_C,ind_CBD_B,ind_CBD_D,\
        ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R,\
        prob_low,prob_high,\
        theta,\
        lambda_minus,lambda_plus)

        # unpack data

        return F

    def gap_rplus(self,Data):
        X    = np.concatenate(([d.flatten()  for d  in Data]),axis=0);
        dFdX = np.concatenate(([df.flatten() for df in F]),   axis=0);

        Y   = np.maximum(0,X - dFdX/self.alpha);
        Z   = X - Y;

        return np.dot(dFdX,Z) - self.alpha/2.*np.dot(Z,Z);

    # Functions used by BloodBank class

    def CalculateNetworkSize(Net):

    nC = Net[0];
    nS = nP = nB = Net[1];
    nD = Net[2];
    nR = Net[3];
    xSize = nC*nB*nD*nR;
    uSize = gamSize = nC+nC*nB+nB+nB+nB*nD+nD*nR;
    NetSize = xSize+uSize+gamSize;

    return NetSize;

    def PathFlow2LinkFlow_x2f(x,\
        alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
        ind_C___C,\
        ind_CB__C,ind_CB__B,\
        ind_CBD_C,ind_CBD_B,ind_CBD_D):

        nC,nB,nD,nR = x.shape;

        f_1C = np.reshape(np.sum(x[ind_C___C,:,:,:],axis=(1,2,3)),(nC,));
        f_CB = np.reshape(np.sum(x[ind_CB__C,ind_CB__B,:,:]*alpha_CBx,axis=(1,2)),(nC,nB));
        f_BP = np.sum(np.reshape(x[ind_CB__C,ind_CB__B,:,:]*alpha_BPx,(nC,nB,nD,nR)),axis=(0,2,3));
        f_PS = f_BP*alpha_PSx;
        f_SD = np.sum(np.reshape(x[ind_CB__C,ind_CB__B,:,:]*alpha_SDx,(nC,nB,nD,nR)),axis=(0,3));
        f_DR = np.sum(np.reshape(x[ind_CBD_C,ind_CBD_B,ind_CBD_D,:]*alpha_DRx,(nC,nB,nD,nR)),axis=(0,1));

        return f_1C, f_CB, f_BP, f_PS, f_SD, f_DR;

    def TotalOperationalCost_Chatx(x,\
        f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
        alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
        chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
        chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
        ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R):

        c_1C = chat_pow2_1C[ind_CBDR_C           ]*f_1C[ind_CBDR_C]           +chat_pow1_1C[ind_CBDR_C];
        c_CB = chat_pow2_CB[ind_CBDR_C,ind_CBDR_B]*f_CB[ind_CBDR_C,ind_CBDR_B]+chat_pow1_CB[ind_CBDR_C,ind_CBDR_B];
        c_BP = chat_pow2_BP[ind_CBDR_B           ]*f_BP[ind_CBDR_B]           +chat_pow1_BP[ind_CBDR_B];
        c_PS = chat_pow2_PS[ind_CBDR_B           ]*f_PS[ind_CBDR_B]           +chat_pow1_PS[ind_CBDR_B];
        c_SD = chat_pow2_SD[ind_CBDR_B,ind_CBDR_D]*f_SD[ind_CBDR_B,ind_CBDR_D]+chat_pow1_SD[ind_CBDR_B,ind_CBDR_D];
        c_DR = chat_pow2_DR[ind_CBDR_D,ind_CBDR_R]*f_DR[ind_CBDR_D,ind_CBDR_R]+chat_pow1_DR[ind_CBDR_D,ind_CBDR_R];

        return x*np.reshape(c_1C+\
                            c_CB*alpha_CBf+\
                            c_BP*alpha_BPf+\
                            c_PS*alpha_PSf+\
                            c_SD*alpha_SDf+\
                            c_DR*alpha_DRf,\
                            x.shape);

    def dTotalOperationalCostdx_dChatxdx(x,\
        f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
        alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
        chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
        chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
        ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R):

        dc_1C = 2.*chat_pow2_1C[ind_CBDR_C           ]*f_1C[ind_CBDR_C]           +chat_pow1_1C[ind_CBDR_C];
        dc_CB = 2.*chat_pow2_CB[ind_CBDR_C,ind_CBDR_B]*f_CB[ind_CBDR_C,ind_CBDR_B]+chat_pow1_CB[ind_CBDR_C,ind_CBDR_B];
        dc_BP = 2.*chat_pow2_BP[ind_CBDR_B           ]*f_BP[ind_CBDR_B]           +chat_pow1_BP[ind_CBDR_B];
        dc_PS = 2.*chat_pow2_PS[ind_CBDR_B           ]*f_PS[ind_CBDR_B]           +chat_pow1_PS[ind_CBDR_B];
        dc_SD = 2.*chat_pow2_SD[ind_CBDR_B,ind_CBDR_D]*f_SD[ind_CBDR_B,ind_CBDR_D]+chat_pow1_SD[ind_CBDR_B,ind_CBDR_D];
        dc_DR = 2.*chat_pow2_DR[ind_CBDR_D,ind_CBDR_R]*f_DR[ind_CBDR_D,ind_CBDR_R]+chat_pow1_DR[ind_CBDR_D,ind_CBDR_R];

        return np.reshape(dc_1C+\
                          dc_CB*alpha_CBf+\
                          dc_BP*alpha_BPf+\
                          dc_PS*alpha_PSf+\
                          dc_SD*alpha_SDf+\
                          dc_DR*alpha_DRf,\
                          x.shape);

    def TotalDiscardingCost_Zhatx(x,\
        f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
        alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
        zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
        ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R):

        z_1C = zhat_1C[ind_CBDR_C           ]*f_1C[ind_CBDR_C];
        z_CB = zhat_CB[ind_CBDR_C,ind_CBDR_B]*f_CB[ind_CBDR_C,ind_CBDR_B];
        z_BP = zhat_BP[ind_CBDR_B           ]*f_BP[ind_CBDR_B];
        z_PS = zhat_PS[ind_CBDR_B           ]*f_PS[ind_CBDR_B];
        z_SD = zhat_SD[ind_CBDR_B,ind_CBDR_D]*f_SD[ind_CBDR_B,ind_CBDR_D];
        z_DR = zhat_DR[ind_CBDR_D,ind_CBDR_R]*f_DR[ind_CBDR_D,ind_CBDR_R];

        return x*np.reshape(z_1C+\
                            z_CB*alpha_CBf+\
                            z_BP*alpha_BPf+\
                            z_PS*alpha_PSf+\
                            z_SD*alpha_SDf+\
                            z_DR*alpha_DRf,\
                            x.shape);

    def dTotalDiscardingCostdx_dZhatxdx(x,\
        f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
        alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
        zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
        ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R):

        dz_1C = 2.*zhat_1C[ind_CBDR_C           ]*f_1C[ind_CBDR_C];
        dz_CB = 2.*zhat_CB[ind_CBDR_C,ind_CBDR_B]*f_CB[ind_CBDR_C,ind_CBDR_B];
        dz_BP = 2.*zhat_BP[ind_CBDR_B           ]*f_BP[ind_CBDR_B];
        dz_PS = 2.*zhat_PS[ind_CBDR_B           ]*f_PS[ind_CBDR_B];
        dz_SD = 2.*zhat_SD[ind_CBDR_B,ind_CBDR_D]*f_SD[ind_CBDR_B,ind_CBDR_D];
        dz_DR = 2.*zhat_DR[ind_CBDR_D,ind_CBDR_R]*f_DR[ind_CBDR_D,ind_CBDR_R];

        return np.reshape(dz_1C+\
                          dz_CB*alpha_CBf+\
                          dz_BP*alpha_BPf+\
                          dz_PS*alpha_PSf+\
                          dz_SD*alpha_SDf+\
                          dz_DR*alpha_DRf,\
                          x.shape);

    def TotalRisk_Rhatx(x,\
        f_1C,\
        rhat,\
        ind_CBDR_C):

        r_1C = rhat[ind_CBDR_C              ]*f_1C[ind_CBDR_C];

        return x*np.reshape(r_1C,\
                            x.shape);

    def dTotalRiskdx_dRhatxdx(x,\
        f_1C,\
        rhat,\
        ind_CBDR_C):

        dr_1C = 2.*rhat[ind_CBDR_C              ]*f_1C[ind_CBDR_C];

        return np.reshape(dr_1C,\
                          x.shape);

    def TotalInvestmentCost_Pihatu(u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
        pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
        pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR):

        pi_1C = pihat_pow2_1C*(u_1C**2)+pihat_pow1_1C*u_1C;
        pi_CB = pihat_pow2_CB*(u_CB**2)+pihat_pow1_CB*u_CB;
        pi_BP = pihat_pow2_BP*(u_BP**2)+pihat_pow1_BP*u_BP;
        pi_PS = pihat_pow2_PS*(u_PS**2)+pihat_pow1_PS*u_PS;
        pi_SD = pihat_pow2_SD*(u_SD**2)+pihat_pow1_SD*u_SD;
        pi_DR = pihat_pow2_DR*(u_DR**2)+pihat_pow1_DR*u_DR;

        return pi_1C, pi_CB, pi_BP, pi_PS, pi_SD, pi_DR;

    def dTotalInvestmentCostdu_dPihatudu(u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
        pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
        pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR):

        dpi_1C = 2.*pihat_pow2_1C*u_1C+pihat_pow1_1C;
        dpi_CB = 2.*pihat_pow2_CB*u_CB+pihat_pow1_CB;
        dpi_BP = 2.*pihat_pow2_BP*u_BP+pihat_pow1_BP;
        dpi_PS = 2.*pihat_pow2_PS*u_PS+pihat_pow1_PS;
        dpi_SD = 2.*pihat_pow2_SD*u_SD+pihat_pow1_SD;
        dpi_DR = 2.*pihat_pow2_DR*u_DR+pihat_pow1_DR;

        return dpi_1C, dpi_CB, dpi_BP, dpi_PS, dpi_SD, dpi_DR;

    def ExpectedShortage_EMinus(x,\
        mu,\
        prob_low,prob_high):

        nu = np.sum(x*mu,axis=(0,1,2));

        return ((nu**2)/2.-nu*prob_high+(prob_high**2)/2.)/(prob_high-prob_low);

    def ExpectedSurplus_EPlus(x,\
        mu,\
        prob_low,prob_high):

        nu = np.sum(x*mu,axis=(0,1,2));

        return ((nu**2)/2.-nu*prob_low+(prob_low**2)/2.)/(prob_high-prob_low);

    def ProbabilityDistributionFunction_Pknuk(x,\
        mu,\
        prob_low,prob_high):

        nu = np.sum(x*mu,axis=(0,1,2));

        return (nu-prob_low)/(prob_high-prob_low);

    def MulticriteriaObjective_Obj(x,\
        u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
        theta,\
        alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
        alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
        chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
        chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
        zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
        pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
        pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR,\
        rhat,\
        ind_C___C,\
        ind_CB__C,ind_CB__B,\
        ind_CBD_C,ind_CBD_B,ind_CBD_D,\
        ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R,\
        mu,\
        prob_low,prob_high,\
        lambda_minus,lambda_plus):

        f_1C, f_CB, f_BP, f_PS, f_SD, f_DR = PathFlow2LinkFlow_x2f(x,\
            alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
            ind_C___C,\
            ind_CB__C,ind_CB__B,\
            ind_CBD_C,ind_CBD_B,ind_CBD_D);
        Chatx = TotalOperationalCost_Chatx(x,\
            f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
            alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
            chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
            chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
            ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R);
        Zhatx = TotalDiscardingCost_Zhatx(x,\
            f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
            alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
            zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
            ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R);
        pi_1C, pi_CB, pi_BP, pi_PS, pi_SD, pi_DR = TotalInvestmentCost_Pihatu(u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
            pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
            pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR);
        EMinus = ExpectedShortage_EMinus(x,\
            mu,\
            prob_low,prob_high);
        # print(EMinus);
        EPlus = ExpectedSurplus_EPlus(x,\
            mu,\
            prob_low,prob_high);
        # print(EPlus);
        Rhatx = TotalRisk_Rhatx(x,\
            f_1C,\
            rhat,\
            ind_CBDR_C);

        # print('Total Investment Cost');
        # print(np.sum(pi_1C)+np.sum(pi_CB)+np.sum(pi_BP)+np.sum(pi_PS)+np.sum(pi_SD)+np.sum(pi_DR));
        # print('Total Objective Cost (17)');
        # print(np.sum(Chatx+Zhatx)+\
        #        np.sum(pi_1C)+np.sum(pi_CB)+np.sum(pi_BP)+np.sum(pi_PS)+np.sum(pi_SD)+np.sum(pi_DR)+\
        #        np.sum(lambda_minus*EMinus+lambda_plus*EPlus));

        return np.sum(Chatx+Zhatx)+\
               np.sum(pi_1C)+np.sum(pi_CB)+np.sum(pi_BP)+np.sum(pi_PS)+np.sum(pi_SD)+np.sum(pi_DR)+\
               np.sum(lambda_minus*EMinus+lambda_plus*EPlus)+\
               np.sum(Rhatx)*theta;

    def Lagrangian_L(x,\
        u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
        gam_1C,gam_CB,gam_BP,gam_PS,gam_SD,gam_DR,\
        ubar_1C,ubar_CB,ubar_BP,ubar_PS,ubar_SD,ubar_DR,\
        theta,\
        alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
        alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
        chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
        chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
        zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
        pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
        pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR,\
        rhat,\
        ind_C___C,\
        ind_CB__C,ind_CB__B,\
        ind_CBD_C,ind_CBD_B,ind_CBD_D,\
        ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R,\
        mu,\
        prob_low,prob_high,\
        lambda_minus,lambda_plus):

        f_1C, f_CB, f_BP, f_PS, f_SD, f_DR = PathFlow2LinkFlow_x2f(x,\
            alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
            ind_C___C,\
            ind_CB__C,ind_CB__B,\
            ind_CBD_C,ind_CBD_B,ind_CBD_D);
        Chatx = TotalOperationalCost_Chatx(x,\
            f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
            alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
            chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
            chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
            ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R);
        Zhatx = TotalDiscardingCost_Zhatx(x,\
            f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
            alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
            zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
            ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R);
        pi_1C, pi_CB, pi_BP, pi_PS, pi_SD, pi_DR = TotalInvestmentCost_Pihatu(u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
            pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
            pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR);
        EMinus = ExpectedShortage_EMinus(x,\
            mu,\
            prob_low,prob_high);
        EPlus = ExpectedSurplus_EPlus(x,\
            mu,\
            prob_low,prob_high);
        Rhatx = TotalRisk_Rhatx(x,\
            f_1C,\
            rhat,\
            ind_CBDR_C);
        dgam_1C, dgam_CB, dgam_BP, dgam_PS, dgam_SD, dgam_DR = F3X_dgam(f_1C, f_CB, f_BP, f_PS, f_SD, f_DR,\
            u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
            ubar_1C,ubar_CB,ubar_BP,ubar_PS,ubar_SD,ubar_DR);

        return np.sum(Chatx+Zhatx)+\
               np.sum(pi_1C)+np.sum(pi_CB)+np.sum(pi_BP)+np.sum(pi_PS)+np.sum(pi_SD)+np.sum(pi_DR)+\
               np.sum(lambda_minus*EMinus+lambda_plus*EPlus)+\
               np.sum(Rhatx)*theta-\
               np.sum(gam_1C*dgam_1C)-np.sum(gam_CB*dgam_CB)-np.sum(gam_BP*dgam_BP)-np.sum(gam_PS*dgam_PS)-np.sum(gam_SD*dgam_SD)-np.sum(gam_DR*dgam_DR);

    def F1X_dx(x,\
        f_1C, f_CB, f_BP, f_PS, f_SD, f_DR,\
        gam_1C,gam_CB,gam_BP,gam_PS,gam_SD,gam_DR,\
        alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
        mu,\
        chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
        chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
        zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
        rhat,\
        ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R,\
        prob_low,prob_high,\
        theta,\
        lambda_minus,lambda_plus):

        nC,nB,nD,nR = x.shape;
        dChatxdx = dTotalOperationalCostdx_dChatxdx(x,\
            f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
            alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
            chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
            chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
            ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R);
        dZhatxdx = dTotalDiscardingCostdx_dZhatxdx(x,\
            f_1C,f_CB,f_BP,f_PS,f_SD,f_DR,\
            alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
            zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
            ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R);
        Pknuk = np.tile(ProbabilityDistributionFunction_Pknuk(x,\
            mu,\
            prob_low,prob_high),\
                        (nC,nB,nD,1));
        Gam = np.reshape(gam_1C[ind_CBDR_C]+\
                         gam_CB[ind_CBDR_C,ind_CBDR_B]+\
                         gam_BP[ind_CBDR_B]+\
                         gam_PS[ind_CBDR_B]+\
                         gam_SD[ind_CBDR_B,ind_CBDR_D]+\
                         gam_DR[ind_CBDR_D,ind_CBDR_R],\
                         (nC,nB,nD,nR));
        dRhatxdx = dTotalRiskdx_dRhatxdx(x,\
            f_1C,\
            rhat,\
            ind_CBDR_C);

        return dChatxdx + dZhatxdx + lambda_plus*mu*Pknuk - lambda_minus*mu*(1-Pknuk) + Gam + theta*dRhatxdx;

    def F2X_du(u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
        gam_1C,gam_CB,gam_BP,gam_PS,gam_SD,gam_DR,\
        pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
        pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR):

        dpi_1C, dpi_CB, dpi_BP, dpi_PS, dpi_SD, dpi_DR = dTotalInvestmentCostdu_dPihatudu(u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
            pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
            pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR);

        du_1C = dpi_1C - gam_1C;
        du_CB = dpi_CB - gam_CB;
        du_BP = dpi_BP - gam_BP;
        du_PS = dpi_PS - gam_PS;
        du_SD = dpi_SD - gam_SD;
        du_DR = dpi_DR - gam_DR;

        return du_1C, du_CB, du_BP, du_PS, du_SD, du_DR;

    def F3X_dgam(f_1C, f_CB, f_BP, f_PS, f_SD, f_DR,\
        u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
        ubar_1C,ubar_CB,ubar_BP,ubar_PS,ubar_SD,ubar_DR):

        dgam_1C = ubar_1C + u_1C - f_1C;
        dgam_CB = ubar_CB + u_CB - f_CB;
        dgam_BP = ubar_BP + u_BP - f_BP;
        dgam_PS = ubar_PS + u_PS - f_PS;
        dgam_SD = ubar_SD + u_SD - f_SD;
        dgam_DR = ubar_DR + u_DR - f_DR;

        return dgam_1C, dgam_CB, dgam_BP, dgam_PS, dgam_SD, dgam_DR;

    def FX_dX(x,\
        u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
        gam_1C,gam_CB,gam_BP,gam_PS,gam_SD,gam_DR,\
        alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
        alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
        mu,\
        chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
        chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
        zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
        rhat,\
        pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
        pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR,\
        ubar_1C,ubar_CB,ubar_BP,ubar_PS,ubar_SD,ubar_DR,\
        ind_C___C,\
        ind_CB__C,ind_CB__B,\
        ind_CBD_C,ind_CBD_B,ind_CBD_D,\
        ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R,\
        prob_low,prob_high,\
        theta,\
        lambda_minus,lambda_plus):

        f_1C, f_CB, f_BP, f_PS, f_SD, f_DR = PathFlow2LinkFlow_x2f(x,\
            alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
            ind_C___C,\
            ind_CB__C,ind_CB__B,\
            ind_CBD_C,ind_CBD_B,ind_CBD_D);
        dx = F1X_dx(x,\
            f_1C, f_CB, f_BP, f_PS, f_SD, f_DR,\
            gam_1C,gam_CB,gam_BP,gam_PS,gam_SD,gam_DR,\
            alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
            mu,\
            chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
            chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
            zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
            rhat,\
            ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R,\
            prob_low,prob_high,\
            theta,\
            lambda_minus,lambda_plus);
        du_1C, du_CB, du_BP, du_PS, du_SD, du_DR = F2X_du(u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
            gam_1C,gam_CB,gam_BP,gam_PS,gam_SD,gam_DR,\
            pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
            pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR);
        dgam_1C, dgam_CB, dgam_BP, dgam_PS, dgam_SD, dgam_DR = F3X_dgam(f_1C, f_CB, f_BP, f_PS, f_SD, f_DR,\
            u_1C,u_CB,u_BP,u_PS,u_SD,u_DR,\
            ubar_1C,ubar_CB,ubar_BP,ubar_PS,ubar_SD,ubar_DR);

        return dx,\
               du_1C,   du_CB,   du_BP,   du_PS,   du_SD,   du_DR,\
               dgam_1C, dgam_CB, dgam_BP, dgam_PS, dgam_SD, dgam_DR;

def CreateNetworkExample1():

    # Example 1 from Nagurney's Supply Chain Network Design of a Sustainable Blood Banking System

    nC = 2;
    nB = 2;
    nP = nB;
    nS = nP;
    nD = 2;
    nR = 3;

    alpha_1C = np.zeros((nC,));
    alpha_CB = np.zeros((nC,nB));
    alpha_BP = np.zeros((nP,));
    alpha_PS = np.zeros((nS,));
    alpha_SD = np.zeros((nS,nD));
    alpha_DR = np.zeros((nD,nR));

    alpha_1C[0]   = .97;
    alpha_1C[1]   = .99;
    alpha_CB[0,0] = 1.;
    alpha_CB[0,1] = .99;
    alpha_CB[1,0] = 1.;
    alpha_CB[1,1] = 1.;
    alpha_BP[0]   = .92;
    alpha_BP[1]   = .96;
    alpha_PS[0]   = .98;
    alpha_PS[1]   = 1.;
    alpha_SD[0,0] = 1.;
    alpha_SD[0,1] = 1.;
    alpha_SD[1,0] = 1.;
    alpha_SD[1,1] = 1.;
    # last set of links has no effect? see formula on page 9 for link flows
    alpha_DR[0,0] = 1.;
    alpha_DR[0,1] = 1.;
    alpha_DR[0,2] = .98;
    alpha_DR[1,0] = 1.;
    alpha_DR[1,1] = 1.;
    alpha_DR[1,2] = .98;

    chat_pow1_1C = np.zeros((nC,));
    chat_pow1_CB = np.zeros((nC,nB));
    chat_pow1_BP = np.zeros((nP,));
    chat_pow1_PS = np.zeros((nS,));
    chat_pow1_SD = np.zeros((nS,nD));
    chat_pow1_DR = np.zeros((nD,nR));

    chat_pow2_1C = np.zeros((nC,));
    chat_pow2_CB = np.zeros((nC,nB));
    chat_pow2_BP = np.zeros((nP,));
    chat_pow2_PS = np.zeros((nS,));
    chat_pow2_SD = np.zeros((nS,nD));
    chat_pow2_DR = np.zeros((nD,nR));

    chat_pow1_1C[0]   = 15.;
    chat_pow2_1C[0]   = 6.;
    chat_pow1_1C[1]   = 11.;
    chat_pow2_1C[1]   = 9.;
    chat_pow1_CB[0,0] = 1.;
    chat_pow2_CB[0,0] = .7;
    chat_pow1_CB[0,1] = 1.;
    chat_pow2_CB[0,1] = 1.2;
    chat_pow1_CB[1,0] = 3.;
    chat_pow2_CB[1,0] = 1.;
    chat_pow1_CB[1,1] = 2.;
    chat_pow2_CB[1,1] = .8;
    chat_pow1_BP[0]   = 2.;
    chat_pow2_BP[0]   = 2.5;
    chat_pow1_BP[1]   = 5.;
    chat_pow2_BP[1]   = 3.;
    chat_pow1_PS[0]   = 6.;
    chat_pow2_PS[0]   = .8;
    chat_pow1_PS[1]   = 3.;
    chat_pow2_PS[1]   = .5;
    chat_pow1_SD[0,0] = 1.;
    chat_pow2_SD[0,0] = .3;
    chat_pow1_SD[0,1] = 2.;
    chat_pow2_SD[0,1] = .5;
    chat_pow1_SD[1,0] = 2.;
    chat_pow2_SD[1,0] = .4;
    chat_pow1_SD[1,1] = 1.;
    chat_pow2_SD[1,1] = .6;
    chat_pow1_DR[0,0] = 1.;
    chat_pow2_DR[0,0] = .4;
    chat_pow1_DR[0,1] = 2.;
    chat_pow2_DR[0,1] = .8;
    chat_pow1_DR[0,2] = 3.;
    chat_pow2_DR[0,2] = .5;
    chat_pow1_DR[1,0] = 1.;
    chat_pow2_DR[1,0] = .7;
    chat_pow1_DR[1,1] = 4.;
    chat_pow2_DR[1,1] = .6;
    chat_pow1_DR[1,2] = 5.;
    chat_pow2_DR[1,2] = 1.1;

    zhat_1C = np.zeros((nC,));
    zhat_CB = np.zeros((nC,nB));
    zhat_BP = np.zeros((nP,));
    zhat_PS = np.zeros((nS,));
    zhat_SD = np.zeros((nS,nD));
    zhat_DR = np.zeros((nD,nR));

    zhat_1C[0]   = .8;
    zhat_1C[1]   = .7;
    zhat_CB[0,0] = .6;
    zhat_CB[0,1] = .8;
    zhat_CB[1,0] = .6;
    zhat_CB[1,1] = .8;
    zhat_BP[0]   = .5;
    zhat_BP[1]   = .8;
    zhat_PS[0]   = .4;
    zhat_PS[1]   = .7;
    zhat_SD[0,0] = .3;
    zhat_SD[0,1] = .4;
    zhat_SD[1,0] = .3;
    zhat_SD[1,1] = .4;
    zhat_DR[0,0] = .7;
    zhat_DR[0,1] = .4;
    zhat_DR[0,2] = .5;
    zhat_DR[1,0] = .7;
    zhat_DR[1,1] = .4;
    zhat_DR[1,2] = .5;

    pihat_pow1_1C = np.zeros((nC,));
    pihat_pow1_CB = np.zeros((nC,nB));
    pihat_pow1_BP = np.zeros((nP,));
    pihat_pow1_PS = np.zeros((nS,));
    pihat_pow1_SD = np.zeros((nS,nD));
    pihat_pow1_DR = np.zeros((nD,nR));

    pihat_pow2_1C = np.zeros((nC,));
    pihat_pow2_CB = np.zeros((nC,nB));
    pihat_pow2_BP = np.zeros((nP,));
    pihat_pow2_PS = np.zeros((nS,));
    pihat_pow2_SD = np.zeros((nS,nD));
    pihat_pow2_DR = np.zeros((nD,nR));

    pihat_pow1_1C[0]   = 1.;
    pihat_pow2_1C[0]   = .8;
    pihat_pow1_1C[1]   = 1.;
    pihat_pow2_1C[1]   = .6;
    pihat_pow1_CB[0,0] = 2.;
    pihat_pow2_CB[0,0] = 1.;
    pihat_pow1_CB[0,1] = 1.;
    pihat_pow2_CB[0,1] = 2.;
    pihat_pow1_CB[1,0] = 1.;
    pihat_pow2_CB[1,0] = 1.;
    pihat_pow1_CB[1,1] = 3.;
    pihat_pow2_CB[1,1] = 1.5;
    pihat_pow1_BP[0]   = 12.;
    pihat_pow2_BP[0]   = 7.;
    pihat_pow1_BP[1]   = 20.;
    pihat_pow2_BP[1]   = 6.;
    pihat_pow1_PS[0]   = 2.;
    pihat_pow2_PS[0]   = 3.;
    pihat_pow1_PS[1]   = 2.;
    pihat_pow2_PS[1]   = 5.4;
    pihat_pow1_SD[0,0] = 1.;
    pihat_pow2_SD[0,0] = 1.;
    pihat_pow1_SD[0,1] = 1.;
    pihat_pow2_SD[0,1] = 1.5;
    pihat_pow1_SD[1,0] = 1.5;
    pihat_pow2_SD[1,0] = 1.8;
    pihat_pow1_SD[1,1] = 2.;
    pihat_pow2_SD[1,1] = 1.;
    pihat_pow1_DR[0,0] = 1.1;
    pihat_pow2_DR[0,0] = .5;
    pihat_pow1_DR[0,1] = 3.;
    pihat_pow2_DR[0,1] = .7;
    pihat_pow1_DR[0,2] = 1.;
    pihat_pow2_DR[0,2] = 2.;
    pihat_pow1_DR[1,0] = 1.;
    pihat_pow2_DR[1,0] = 1.;
    pihat_pow1_DR[1,1] = 2.;
    pihat_pow2_DR[1,1] = 1.;
    pihat_pow1_DR[1,2] = 1.;
    pihat_pow2_DR[1,2] = .8;

    prob_low  = np.zeros((nR,));
    prob_high = np.zeros((nR,));

    prob_low[0]  = 5.;
    prob_high[0] = 10.;
    prob_low[1]  = 40.;
    prob_high[1] = 50.;
    prob_low[2]  = 25.;
    prob_high[2] = 40.;

    lambda_minus = np.zeros((nR,));
    lambda_plus  = np.zeros((nR,));

    lambda_minus[0] = 2800.;
    lambda_plus[0]  = 50.;
    lambda_minus[1] = 3000.;
    lambda_plus[1]  = 60.;
    lambda_minus[2] = 3100.;
    lambda_plus[2]  = 50.;

    rhat = np.zeros((nC,));

    rhat[0] = 2.;
    rhat[1] = 1.5;

    ubar_1C = np.zeros((nC,));
    ubar_CB = np.zeros((nC,nB));
    ubar_BP = np.zeros((nB,));
    ubar_PS = np.zeros((nB,));
    ubar_SD = np.zeros((nB,nD));
    ubar_DR = np.zeros((nD,nR));

    theta = .7;

    #Helper Arguments

    #Index Lists for Fast Slicing
    ind_C___C = np.arange(nC);
    ind_CB__C = np.tile(np.arange(nC),(nB,1)).T.flatten();
    ind_CB__B = np.tile(np.arange(nB),nC);
    ind_CBD_C = np.tile(np.arange(nC),(nB*nD,1)).T.flatten();
    ind_CBD_B = np.tile(np.tile(np.arange(nB),(nD,1)).T.flatten(),nC);
    ind_CBD_D = np.tile(np.arange(nD),nC*nB);
    ind_CBDR_C = np.tile(np.arange(nC),(nB*nD*nR,1)).T.flatten();
    ind_CBDR_B = np.tile(np.tile(np.arange(nB),(nD*nR,1)).T.flatten(),nC);
    ind_CBDR_D = np.tile(np.tile(np.arange(nD),(nR,1)).T.flatten(),nC*nB);
    ind_CBDR_R = np.tile(np.arange(nR),nC*nB*nD);

    #Alpha Values for Both Path and Link Computation
    alpha_CBx = np.tile(np.rollaxis(alpha_1C[ind_CB__C][None][None],2),(1,nD,nR));
    alpha_BPx = np.tile(np.rollaxis(alpha_1C[ind_CB__C][None][None],2),(1,nD,nR))*\
                np.tile(np.rollaxis(alpha_CB[ind_CB__C,ind_CB__B][None][None],2),(1,nD,nR));
    alpha_PSx = alpha_BP;
    alpha_SDx = np.tile(np.rollaxis(alpha_1C[ind_CB__C][None][None],2),(1,nD,nR))*\
                np.tile(np.rollaxis(alpha_CB[ind_CB__C,ind_CB__B][None][None],2),(1,nD,nR))*\
                np.tile(np.rollaxis(alpha_BP[ind_CB__B][None][None],2),(1,nD,nR))*\
                np.tile(np.rollaxis(alpha_PS[ind_CB__B][None][None],2),(1,nD,nR));
    alpha_DRx = np.tile(np.rollaxis(alpha_1C[ind_CBD_C][None],1),(1,nR))*\
                np.tile(np.rollaxis(alpha_CB[ind_CBD_C,ind_CBD_B][None],1),(1,nR))*\
                np.tile(np.rollaxis(alpha_BP[ind_CBD_B][None],1),(1,nR))*\
                np.tile(np.rollaxis(alpha_PS[ind_CBD_B][None],1),(1,nR))*\
                np.tile(np.rollaxis(alpha_SD[ind_CBD_B,ind_CBD_D][None],1),(1,nR));
    alpha_CBf  = alpha_1C[ind_CBDR_C];
    alpha_BPf  = alpha_CBf*alpha_CB[ind_CBDR_C,ind_CBDR_B];
    alpha_PSf  = alpha_BPf*alpha_BP[ind_CBDR_B];
    alpha_SDf  = alpha_PSf*alpha_PS[ind_CBDR_B];
    alpha_DRf  = alpha_SDf*alpha_SD[ind_CBDR_B,ind_CBDR_D];

    #Mu
    mu = np.reshape(alpha_DRf*alpha_DR[ind_CBDR_D,ind_CBDR_R],(nC,nB,nD,nR));

    # return network as package or list (not individual variables)

    return alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
           alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
           mu,\
           chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
           chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
           zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
           rhat,\
           pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
           pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR,\
           ubar_1C,ubar_CB,ubar_BP,ubar_PS,ubar_SD,ubar_DR,\
           ind_C___C,\
           ind_CB__C,ind_CB__B,\
           ind_CBD_C,ind_CBD_B,ind_CBD_D,\
           ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R,\
           prob_low,prob_high,\
           theta,\
           lambda_minus,lambda_plus;

def CreateRandomNetwork(nC,nB,nD,nR,seed):

    np.random.seed(seed);

    nP = nS = nB;

    alpha_1C = .9+.1*np.random.rand(nC);
    alpha_CB = .9+.1*np.random.rand(nC,nB);
    alpha_BP = .9+.1*np.random.rand(nP);
    alpha_PS = .9+.1*np.random.rand(nS);
    alpha_SD = .9+.1*np.random.rand(nS,nD);
    alpha_DR = .9+.1*np.random.rand(nD,nR);

    chat_pow1_1C = 10.+10.*np.random.rand(nC);
    chat_pow1_CB = .5 +3.5*np.random.rand(nC,nB);
    chat_pow1_BP = 1. + 5.*np.random.rand(nP);
    chat_pow1_PS = 2. + 8.*np.random.rand(nS);
    chat_pow1_SD = .5 + 2.*np.random.rand(nS,nD);
    chat_pow1_DR = .5 + 6.*np.random.rand(nD,nR);

    chat_pow2_1C = 5. +5.*np.random.rand(nC);
    chat_pow2_CB = .5 +1.*np.random.rand(nC,nB);
    chat_pow2_BP = 1. +3.*np.random.rand(nP);
    chat_pow2_PS = .5 +.5*np.random.rand(nS);
    chat_pow2_SD = 0. +1.*np.random.rand(nS,nD);
    chat_pow2_DR = .3 +1.*np.random.rand(nD,nR);

    zhat_1C = .6+.3*np.random.rand(nC);
    zhat_CB = .5+.4*np.random.rand(nC,nB);
    zhat_BP = .4+.5*np.random.rand(nP);
    zhat_PS = .3+.5*np.random.rand(nS);
    zhat_SD = .2+.3*np.random.rand(nS,nD);
    zhat_DR = .3+.6*np.random.rand(nD,nR);

    pihat_pow1_1C = .5 +  1.*np.random.rand(nC);
    pihat_pow1_CB = .5 +  3.*np.random.rand(nC,nB);
    pihat_pow1_BP = 10.+ 15.*np.random.rand(nP);
    pihat_pow1_PS = 1. +  2.*np.random.rand(nS);
    pihat_pow1_SD = .5 +  2.*np.random.rand(nS,nD);
    pihat_pow1_DR = .8 + 2.5*np.random.rand(nD,nR);

    pihat_pow2_1C = .5+.5*np.random.rand(nC);
    pihat_pow2_CB = .5+2.*np.random.rand(nC,nB);
    pihat_pow2_BP = 5.+5.*np.random.rand(nP);
    pihat_pow2_PS = 2.+6.*np.random.rand(nS);
    pihat_pow2_SD = .8+2.*np.random.rand(nS,nD);
    pihat_pow2_DR = .4+2.*np.random.rand(nD,nR);

    prob_low  = 2.+48.*np.random.rand(nR);
    prob_high = 3.+17.*np.random.rand(nR)+prob_low;

    lambda_minus = 2500.+1000.*np.random.rand(nR);
    lambda_plus  = 25.  +  50.*np.random.rand(nR);

    rhat = 1.+1.5*np.random.rand(nC);

    #all random networks are resdesigns from scratch
    ubar_1C = np.zeros((nC,));
    ubar_CB = np.zeros((nC,nB));
    ubar_BP = np.zeros((nB,));
    ubar_PS = np.zeros((nB,));
    ubar_SD = np.zeros((nB,nD));
    ubar_DR = np.zeros((nD,nR));

    theta = .5+.5*np.random.rand();

    #Helper Arguments

    #Index Lists for Fast Slicing
    ind_C___C = np.arange(nC);
    ind_CB__C = np.tile(np.arange(nC),(nB,1)).T.flatten();
    ind_CB__B = np.tile(np.arange(nB),nC);
    ind_CBD_C = np.tile(np.arange(nC),(nB*nD,1)).T.flatten();
    ind_CBD_B = np.tile(np.tile(np.arange(nB),(nD,1)).T.flatten(),nC);
    ind_CBD_D = np.tile(np.arange(nD),nC*nB);
    ind_CBDR_C = np.tile(np.arange(nC),(nB*nD*nR,1)).T.flatten();
    ind_CBDR_B = np.tile(np.tile(np.arange(nB),(nD*nR,1)).T.flatten(),nC);
    ind_CBDR_D = np.tile(np.tile(np.arange(nD),(nR,1)).T.flatten(),nC*nB);
    ind_CBDR_R = np.tile(np.arange(nR),nC*nB*nD);

    #Alpha Values for Both Path and Link Computation
    alpha_CBx = np.tile(np.rollaxis(alpha_1C[ind_CB__C][None][None],2),(1,nD,nR));
    alpha_BPx = np.tile(np.rollaxis(alpha_1C[ind_CB__C][None][None],2),(1,nD,nR))*\
                np.tile(np.rollaxis(alpha_CB[ind_CB__C,ind_CB__B][None][None],2),(1,nD,nR));
    alpha_PSx = alpha_BP;
    alpha_SDx = np.tile(np.rollaxis(alpha_1C[ind_CB__C][None][None],2),(1,nD,nR))*\
                np.tile(np.rollaxis(alpha_CB[ind_CB__C,ind_CB__B][None][None],2),(1,nD,nR))*\
                np.tile(np.rollaxis(alpha_BP[ind_CB__B][None][None],2),(1,nD,nR))*\
                np.tile(np.rollaxis(alpha_PS[ind_CB__B][None][None],2),(1,nD,nR));
    alpha_DRx = np.tile(np.rollaxis(alpha_1C[ind_CBD_C][None],1),(1,nR))*\
                np.tile(np.rollaxis(alpha_CB[ind_CBD_C,ind_CBD_B][None],1),(1,nR))*\
                np.tile(np.rollaxis(alpha_BP[ind_CBD_B][None],1),(1,nR))*\
                np.tile(np.rollaxis(alpha_PS[ind_CBD_B][None],1),(1,nR))*\
                np.tile(np.rollaxis(alpha_SD[ind_CBD_B,ind_CBD_D][None],1),(1,nR));
    alpha_CBf  = alpha_1C[ind_CBDR_C];
    alpha_BPf  = alpha_CBf*alpha_CB[ind_CBDR_C,ind_CBDR_B];
    alpha_PSf  = alpha_BPf*alpha_BP[ind_CBDR_B];
    alpha_SDf  = alpha_PSf*alpha_PS[ind_CBDR_B];
    alpha_DRf  = alpha_SDf*alpha_SD[ind_CBDR_B,ind_CBDR_D];

    #Mu
    mu = np.reshape(alpha_DRf*alpha_DR[ind_CBDR_D,ind_CBDR_R],(nC,nB,nD,nR));

    # return network as package or list (not individual variables)

    return alpha_CBx,alpha_BPx,alpha_PSx,alpha_SDx,alpha_DRx,\
           alpha_CBf,alpha_BPf,alpha_PSf,alpha_SDf,alpha_DRf,\
           mu,\
           chat_pow1_1C,chat_pow1_CB,chat_pow1_BP,chat_pow1_PS,chat_pow1_SD,chat_pow1_DR,\
           chat_pow2_1C,chat_pow2_CB,chat_pow2_BP,chat_pow2_PS,chat_pow2_SD,chat_pow2_DR,\
           zhat_1C,zhat_CB,zhat_BP,zhat_PS,zhat_SD,zhat_DR,\
           rhat,\
           pihat_pow1_1C,pihat_pow1_CB,pihat_pow1_BP,pihat_pow1_PS,pihat_pow1_SD,pihat_pow1_DR,\
           pihat_pow2_1C,pihat_pow2_CB,pihat_pow2_BP,pihat_pow2_PS,pihat_pow2_SD,pihat_pow2_DR,\
           ubar_1C,ubar_CB,ubar_BP,ubar_PS,ubar_SD,ubar_DR,\
           ind_C___C,\
           ind_CB__C,ind_CB__B,\
           ind_CBD_C,ind_CBD_B,ind_CBD_D,\
           ind_CBDR_C,ind_CBDR_B,ind_CBDR_D,ind_CBDR_R,\
           prob_low,prob_high,\
           theta,\
           lambda_minus,lambda_plus;