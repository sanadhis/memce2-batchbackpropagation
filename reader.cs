using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MySql.Data.MySqlClient;

using System.Data;
using MySql.Data.Common;
using MySql.Data.Types;

public class Reader
{

    static void Main()
    {
        string cs = @"server=localhost;Uid=root;
            Pwd=root;database=memce2";

        MySqlConnection conn = null;

        try
        {
            conn = new MySqlConnection(cs);
            conn.Open();
            Console.WriteLine("MySQL version : {0}", conn.ServerVersion);

        }
        catch (MySqlException ex)
        {
            Console.WriteLine("Error: {0}", ex.ToString());

        }
        finally
        {
            if (conn != null)
            {
                MySqlCommand mSqlCmdSelectCustomers = conn.CreateCommand();
                mSqlCmdSelectCustomers.CommandText = @"select * from ambildata";
                MySqlDataReader mSqlReader_Customers;
                mSqlReader_Customers = mSqlCmdSelectCustomers.ExecuteReader();
                while (mSqlReader_Customers.Read())
                {
                    string nama = mSqlReader_Customers.GetString(1);
                    Console.WriteLine(nama);
                }
            }
        }      
        Console.ReadLine();
    }
}