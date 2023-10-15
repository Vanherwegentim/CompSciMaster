package hotel;

import java.rmi.Remote;
import java.rmi.RemoteException;

public interface BookingManagerRemote extends Remote {
    void PrintMsg() throws RemoteException;
}
